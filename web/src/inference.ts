import { Config, Vocab, PunctuationMap } from './types'

const defaultConfig: Config = {
  model: {
    vocabSize: 8000,
    embedDim: 128,
    hiddenDim: 256,
    numLayers: 4,
    numHeads: 8,
    numLabels: 5,
    maxSeqLen: 256
  },
  punctuationMap: {
    O: 0,
    COMMA: 1,
    PERIOD: 2,
    QUESTION: 3,
    EXCLAMATION: 4
  },
  punctuationTokens: {
    O: '',
    COMMA: '，',
    PERIOD: '。',
    QUESTION: '？',
    EXCLAMATION: '！'
  },
  chinesePunctuation: ['，', '。', '？', '！', '、', '；', '：', '\u201c', '\u201d', '\u2018', '\u2019', '（', '）', '【', '】', '《', '》', ',', '!', '?', '.', ';', ':', '(', ')', '[', ']']
}

export class PunctuationInference {
  private session: Awaited<ReturnType<typeof import('onnxruntime-web').InferenceSession.create>> | null = null
  private vocab: Vocab = {}
  private config: Config = defaultConfig
  private loaded = false

  async loadModel(modelPath: string, vocabPath: string): Promise<void> {
    const ort = await import('onnxruntime-web')

    const [session, vocabResp] = await Promise.all([
      ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm']
      }),
      fetch(vocabPath).then(r => r.json()) as Promise<Vocab>
    ])
    
    this.session = session
    this.vocab = vocabResp
    this.loaded = true
  }

  private cleanText(text: string): string {
    return text.split('').filter(c => !this.config.chinesePunctuation.includes(c)).join('')
  }

  private textToIds(text: string): number[] {
    const cleanText = this.cleanText(text)
    const unkId = this.vocab['<UNK>'] ?? 1
    return cleanText.split('').map(char => this.vocab[char] ?? unkId)
  }

  isLoaded(): boolean {
    return this.loaded
  }

  async predict(text: string): Promise<string> {
    if (!this.session) {
      throw new Error('Model not loaded')
    }

    const ort = await import('onnxruntime-web')
    const cleanText = this.cleanText(text)
    const charIds = this.textToIds(text)
    
    if (charIds.length > this.config.model.maxSeqLen) {
      charIds.splice(this.config.model.maxSeqLen)
    }

    const inputIds = new ort.Tensor('int64', BigInt64Array.from(charIds.map(BigInt)), [1, charIds.length])
    const attentionMask = new ort.Tensor('int64', BigInt64Array.from(new Array(charIds.length).fill(BigInt(1))), [1, charIds.length])

    const results = await this.session.run({
      input_ids: inputIds,
      attention_mask: attentionMask
    })

    const logits = results.logits.data as Float32Array
    const preds: number[] = []
    
    for (let i = 0; i < charIds.length; i++) {
      let maxIdx = 0
      let maxVal = -Infinity
      for (let j = 0; j < this.config.model.numLabels; j++) {
        const val = logits[i * this.config.model.numLabels + j]
        if (val > maxVal) {
          maxVal = val
          maxIdx = j
        }
      }
      preds.push(maxIdx)
    }

    return this.decode(cleanText.slice(0, preds.length), preds)
  }

  private decode(text: string, preds: number[]): string {
    const punctuationTokens = this.config.punctuationTokens
    const punctuationMap = this.config.punctuationMap
    
    const reverseMap: Record<number, keyof PunctuationMap> = {}
    for (const [key, value] of Object.entries(punctuationMap) as [keyof PunctuationMap, number][]) {
      reverseMap[value] = key
    }

    const result: string[] = []
    for (let i = 0; i < text.length; i++) {
      result.push(text[i])
      const label = preds[i]
      const punctName = reverseMap[label]
      if (punctName && punctName !== 'O') {
        result.push(punctuationTokens[punctName])
      }
    }

    return result.join('')
  }
}

export const inference = new PunctuationInference()