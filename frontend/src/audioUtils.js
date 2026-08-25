/**
 * WAV 编码工具 — 浏览器端 Float32Array → WAV Blob
 * 仅依赖 Web 标准 API，无外部依赖
 */

function floatTo16BitPCM(input) {
  const output = new Int16Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]))
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return output
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i))
  }
}

export function encodeWav(float32Array, sampleRate, numChannels = 1) {
  const bitsPerSample = 16
  const blockAlign = numChannels * bitsPerSample / 8
  const byteRate = sampleRate * blockAlign
  const dataSize = float32Array.length * 2 // 16-bit = 2 bytes
  const chunkSize = 36 + dataSize

  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  // RIFF header
  writeString(view, 0, 'RIFF')
  view.setUint32(4, chunkSize, true)
  writeString(view, 8, 'WAVE')

  // fmt chunk
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)           // chunk size (16 for PCM)
  view.setUint16(20, 1, true)            // audio format (1 = PCM)
  view.setUint16(22, numChannels, true)  // channels
  view.setUint32(24, sampleRate, true)   // sample rate
  view.setUint32(28, byteRate, true)     // byte rate
  view.setUint16(32, blockAlign, true)   // block align
  view.setUint16(34, bitsPerSample, true)// bits per sample

  // data chunk
  writeString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  // PCM data
  const pcm = floatTo16BitPCM(float32Array)
  const dataView = new DataView(buffer, 44)
  for (let i = 0; i < pcm.length; i++) {
    dataView.setInt16(i * 2, pcm[i], true)
  }

  return new Blob([buffer], { type: 'audio/wav' })
}