import { startCamera } from './camera.ts'

const video = document.querySelector<HTMLVideoElement>('#feed')!
const message = document.querySelector<HTMLDivElement>('#message')!

function showMessage(text: string): void {
  message.textContent = text
  message.classList.remove('hidden')
}

async function main(): Promise<void> {
  try {
    const stream = await startCamera()
    video.srcObject = stream
  } catch (error) {
    const name = error instanceof Error ? error.name : ''
    switch (name) {
      case 'NotAllowedError':
      case 'SecurityError':
        showMessage('Camera access denied. Please allow camera permission and reload.')
        break
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        showMessage('No camera found.')
        break
      default:
        showMessage(`Could not start camera${name ? ` (${name})` : ''}.`)
    }
  }
}

main()
