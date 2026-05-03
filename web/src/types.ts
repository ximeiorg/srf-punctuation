export interface Vocab {
  [char: string]: number
}

export interface PunctuationMap {
  O: number
  COMMA: number
  PERIOD: number
  QUESTION: number
  EXCLAMATION: number
}

export interface PunctuationTokens {
  O: string
  COMMA: string
  PERIOD: string
  QUESTION: string
  EXCLAMATION: string
}

export interface ModelConfig {
  vocabSize: number
  embedDim: number
  hiddenDim: number
  numLayers: number
  numHeads: number
  numLabels: number
  maxSeqLen: number
}

export interface Config {
  model: ModelConfig
  punctuationMap: PunctuationMap
  punctuationTokens: PunctuationTokens
  chinesePunctuation: string[]
}