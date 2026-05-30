/**
 * Request the webcam and return its MediaStream.
 *
 * No DOM access and no global state: the stream is handed back to the caller so
 * future consumers (e.g. an ASL recognition loop) can share the same feed
 * without this module needing to change.
 */
export function startCamera(
  constraints: MediaStreamConstraints = { video: true, audio: false },
): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    return Promise.reject(new Error('getUserMedia is not supported in this browser'))
  }
  return navigator.mediaDevices.getUserMedia(constraints)
}
