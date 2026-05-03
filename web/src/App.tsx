import { useState, useEffect, useCallback } from 'react'
import { inference } from './inference'
import './App.css'

const MODEL_PATH = '/punctuation_int8.onnx'
const VOCAB_PATH = '/vocab.json'

const EXAMPLES = [
  '今天天气很好我们出去散步吧',
  '你吃饭了吗我还没吃呢',
  '请问这个多少钱能不能便宜点',
  '太棒了这个产品真的非常好用',
  '明天早上八点开会不要迟到',
  '我觉得这个方案可行但是需要一些调整',
  '你是谁为什么会在这里',
  '好的我知道了马上就去办'
]

function App() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [errorMessage, setErrorMessage] = useState('')
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    inference.loadModel(MODEL_PATH, VOCAB_PATH)
      .then(() => setStatus('ready'))
      .catch(err => {
        console.error(err)
        setStatus('error')
        setErrorMessage(err.message || '模型加载失败')
      })
  }, [])

  const handlePredict = useCallback(async () => {
    if (!inputText.trim() || processing) return
    
    setProcessing(true)
    try {
      const result = await inference.predict(inputText)
      setOutputText(result)
    } catch (err) {
      console.error(err)
      setOutputText('预测失败: ' + (err as Error).message)
    } finally {
      setProcessing(false)
    }
  }, [inputText, processing])

  const handleClear = useCallback(() => {
    setInputText('')
    setOutputText('')
  }, [])

  const handleExample = useCallback((text: string) => {
    setInputText(text)
    setOutputText('')
  }, [])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handlePredict()
    }
  }, [handlePredict])

  return (
    <div className="app">
      <h1>语音文本标点预测</h1>
      
      <div className={`status ${status}`}>
        {status === 'loading' && '正在加载模型...'}
        {status === 'ready' && '模型已就绪'}
        {status === 'error' && `加载失败: ${errorMessage}`}
      </div>

      {status === 'ready' && (
        <>
          <div className="input-section">
            <textarea
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入不带标点的文本，或输入有错误标点的文本..."
              disabled={processing}
            />
            <div className="buttons">
              <button 
                className="primary" 
                onClick={handlePredict}
                disabled={!inputText.trim() || processing}
              >
                {processing ? '处理中...' : '预测标点'}
              </button>
              <button 
                className="secondary" 
                onClick={handleClear}
                disabled={processing}
              >
                清空
              </button>
            </div>
          </div>

          <div className="output-section">
            <div className="output-label">预测结果：</div>
            <div className={`output-box ${!outputText ? 'empty' : ''}`}>
              {outputText || '预测结果将显示在这里'}
            </div>
          </div>

          <div className="examples">
            <div className="examples-label">示例文本：</div>
            <div className="example-buttons">
              {EXAMPLES.map((text, i) => (
                <button
                  key={i}
                  className="example"
                  onClick={() => handleExample(text)}
                  disabled={processing}
                >
                  {text.slice(0, 10)}...
                </button>
              ))}
            </div>
          </div>

          <div className="info">
            提示：按 Ctrl+Enter 快速预测。模型会自动清除输入中的标点并重新预测。
          </div>
        </>
      )}
    </div>
  )
}

export default App