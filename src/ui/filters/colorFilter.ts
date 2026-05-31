import { coverParams } from '../coverFit.ts'

/**
 * WebGL color-grade engine.
 *
 * Every "color filter" is a {@link ColorFilter} parameter set fed to one
 * comprehensive grading shader, so we ship a large catalogue of polished looks
 * (see ./presets.ts) without writing any new GL.
 *
 * The pass renders the live camera through the grade onto an opaque canvas that
 * sits directly above the (hidden-behind) `<video>` and below the lens overlay —
 * matching the video's mirror + `object-fit: cover`. When no filter
 * is active the canvas is hidden and the raw video shows through. If WebGL is
 * unavailable, {@link createColorFilterRenderer} returns null.
 */

export interface Duotone {
  dark: [number, number, number]
  light: [number, number, number]
  mix: number
}

/** A color look. Every field is optional; omitted fields are neutral. */
export interface ColorFilter {
  exposure?: number
  contrast?: number
  saturation?: number
  vibrance?: number
  temperature?: number
  tint?: number
  hue?: number
  gamma?: number
  fade?: number
  lift?: [number, number, number]
  gain?: [number, number, number]
  vignette?: number
  vignetteSoftness?: number
  grain?: number
  glow?: number
  sharpen?: number
  posterize?: number
  scanline?: number
  duotone?: Duotone
}

const VERT = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`

const FRAG = `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uVisible;     // cover-fit visible source fraction
uniform vec2 uTexel;       // 1/resolution
uniform float uTime;

uniform float uExposure, uContrast, uSaturation, uVibrance;
uniform float uTemperature, uTint, uHue, uGamma, uFade;
uniform vec3 uLift, uGain;
uniform float uVignette, uVignetteSoft;
uniform float uGrain, uGlow, uSharpen, uPosterize, uScanline;
uniform vec3 uDuoDark, uDuoLight;
uniform float uDuotone;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 345.45));
  p += dot(p, p + 34.345);
  return fract(p.x * p.y);
}
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 sourceUv(vec2 uv) {
  vec2 c = (uv - 0.5) * uVisible + 0.5;
  c.x = 1.0 - c.x;
  return c;
}

void main() {
  vec2 suv = sourceUv(vUv);
  vec3 col = texture2D(uTex, suv).rgb;

  if (uSharpen > 0.0) {
    vec3 blur =
      texture2D(uTex, sourceUv(vUv + vec2(uTexel.x, 0.0))).rgb +
      texture2D(uTex, sourceUv(vUv - vec2(uTexel.x, 0.0))).rgb +
      texture2D(uTex, sourceUv(vUv + vec2(0.0, uTexel.y))).rgb +
      texture2D(uTex, sourceUv(vUv - vec2(0.0, uTexel.y))).rgb;
    col += (col - blur * 0.25) * uSharpen;
  }
  if (uGlow > 0.0) {
    vec3 g = vec3(0.0);
    for (int i = 0; i < 8; i++) {
      float a = float(i) / 8.0 * 6.2831853;
      vec2 off = vec2(cos(a), sin(a)) * uTexel * 4.0;
      g += texture2D(uTex, sourceUv(vUv + off)).rgb;
    }
    col += max(g / 8.0 - 0.6, 0.0) * uGlow * 2.0;
  }

  col *= exp2(uExposure);
  col.r += uTemperature * 0.10;
  col.b -= uTemperature * 0.10;
  col.g += uTint * 0.10;
  col = (col - 0.5) * uContrast + 0.5;

  float l = dot(col, LUMA);
  col = mix(vec3(l), col, uSaturation);
  if (uVibrance != 0.0) {
    float sat = max(max(col.r, col.g), col.b) - min(min(col.r, col.g), col.b);
    col = mix(vec3(l), col, 1.0 + uVibrance * (1.0 - sat));
  }
  if (uHue != 0.0) {
    vec3 hsv = rgb2hsv(max(col, 0.0));
    hsv.x = fract(hsv.x + uHue / 360.0);
    col = hsv2rgb(hsv);
  }
  col = max(col, 0.0);
  col = pow(col, vec3(1.0 / uGamma));

  float lum = dot(col, LUMA);
  col += uLift * (1.0 - lum);
  col *= (vec3(1.0) + uGain * lum);
  col = mix(col, max(col, vec3(uFade)), step(0.0001, uFade));

  if (uDuotone > 0.0) {
    float t = dot(clamp(col, 0.0, 1.0), LUMA);
    col = mix(col, mix(uDuoDark, uDuoLight, t), uDuotone);
  }
  if (uPosterize > 0.0) col = floor(col * uPosterize) / uPosterize;
  col = clamp(col, 0.0, 1.0);

  if (uVignette > 0.0) {
    float d = distance(vUv, vec2(0.5));
    float v = smoothstep(0.75, 0.75 - max(uVignetteSoft, 0.05), d);
    col *= mix(1.0, v, uVignette);
  }
  if (uScanline > 0.0) {
    float sln = 0.5 + 0.5 * sin(vUv.y / uTexel.y * 3.14159);
    col *= 1.0 - uScanline * (1.0 - sln);
  }
  if (uGrain > 0.0) {
    col += (hash(vUv * vec2(1920.0, 1080.0) + uTime) - 0.5) * uGrain * 0.18;
  }

  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`

type UniformName =
  | 'uVisible' | 'uTexel' | 'uTime'
  | 'uExposure' | 'uContrast' | 'uSaturation' | 'uVibrance'
  | 'uTemperature' | 'uTint' | 'uHue' | 'uGamma' | 'uFade'
  | 'uLift' | 'uGain' | 'uVignette' | 'uVignetteSoft'
  | 'uGrain' | 'uGlow' | 'uSharpen' | 'uPosterize' | 'uScanline'
  | 'uDuoDark' | 'uDuoLight' | 'uDuotone'

export class ColorFilterRenderer {
  private canvas: HTMLCanvasElement
  private video: HTMLVideoElement
  private gl: WebGLRenderingContext
  private program: WebGLProgram
  private tex: WebGLTexture
  private u: Record<UniformName, WebGLUniformLocation | null>
  private filter: ColorFilter | null = null
  private rafId = 0
  private start = performance.now()

  constructor(canvas: HTMLCanvasElement, video: HTMLVideoElement, gl: WebGLRenderingContext) {
    this.canvas = canvas
    this.video = video
    this.gl = gl
    this.program = buildProgram(gl)
    this.tex = createTexture(gl)

    const quad = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, quad)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)
    const aPos = gl.getAttribLocation(this.program, 'aPos')
    gl.enableVertexAttribArray(aPos)
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0)

    const names: UniformName[] = [
      'uVisible', 'uTexel', 'uTime', 'uExposure', 'uContrast', 'uSaturation',
      'uVibrance', 'uTemperature', 'uTint', 'uHue', 'uGamma', 'uFade', 'uLift',
      'uGain', 'uVignette', 'uVignetteSoft', 'uGrain', 'uGlow', 'uSharpen',
      'uPosterize', 'uScanline', 'uDuoDark', 'uDuoLight', 'uDuotone',
    ]
    this.u = {} as Record<UniformName, WebGLUniformLocation | null>
    gl.useProgram(this.program)
    for (const n of names) this.u[n] = gl.getUniformLocation(this.program, n)
    gl.uniform1i(gl.getUniformLocation(this.program, 'uTex'), 0)

    this.resize()
    window.addEventListener('resize', () => this.resize())
  }

  /** null clears the filter (canvas hidden, raw video shows through). */
  setFilter(filter: ColorFilter | null): void {
    this.filter = filter
    this.canvas.style.display = filter ? 'block' : 'none'
  }

  startLoop(): void {
    if (this.rafId) return
    const loop = (): void => {
      this.rafId = requestAnimationFrame(loop)
      this.draw()
    }
    loop()
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.rafId = 0
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    this.canvas.width = Math.round(window.innerWidth * dpr)
    this.canvas.height = Math.round(window.innerHeight * dpr)
  }

  private draw(): void {
    const f = this.filter
    if (!f || this.video.readyState < 2) return
    const gl = this.gl
    gl.viewport(0, 0, this.canvas.width, this.canvas.height)
    gl.useProgram(this.program)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.tex)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.video)

    const cover = coverParams(this.video)
    gl.uniform2f(this.u.uVisible, cover.width / (cover.videoW * cover.scale), cover.height / (cover.videoH * cover.scale))
    gl.uniform2f(this.u.uTexel, 1 / this.canvas.width, 1 / this.canvas.height)
    gl.uniform1f(this.u.uTime, (performance.now() - this.start) % 1000)

    gl.uniform1f(this.u.uExposure, f.exposure ?? 0)
    gl.uniform1f(this.u.uContrast, f.contrast ?? 1)
    gl.uniform1f(this.u.uSaturation, f.saturation ?? 1)
    gl.uniform1f(this.u.uVibrance, f.vibrance ?? 0)
    gl.uniform1f(this.u.uTemperature, f.temperature ?? 0)
    gl.uniform1f(this.u.uTint, f.tint ?? 0)
    gl.uniform1f(this.u.uHue, f.hue ?? 0)
    gl.uniform1f(this.u.uGamma, f.gamma ?? 1)
    gl.uniform1f(this.u.uFade, f.fade ?? 0)
    gl.uniform3fv(this.u.uLift, f.lift ?? [0, 0, 0])
    gl.uniform3fv(this.u.uGain, f.gain ?? [0, 0, 0])
    gl.uniform1f(this.u.uVignette, f.vignette ?? 0)
    gl.uniform1f(this.u.uVignetteSoft, f.vignetteSoftness ?? 0.3)
    gl.uniform1f(this.u.uGrain, f.grain ?? 0)
    gl.uniform1f(this.u.uGlow, f.glow ?? 0)
    gl.uniform1f(this.u.uSharpen, f.sharpen ?? 0)
    gl.uniform1f(this.u.uPosterize, f.posterize ?? 0)
    gl.uniform1f(this.u.uScanline, f.scanline ?? 0)
    gl.uniform3fv(this.u.uDuoDark, f.duotone?.dark ?? [0, 0, 0])
    gl.uniform3fv(this.u.uDuoLight, f.duotone?.light ?? [1, 1, 1])
    gl.uniform1f(this.u.uDuotone, f.duotone?.mix ?? 0)

    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }
}

export function createColorFilterRenderer(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
): ColorFilterRenderer | null {
  const gl = (canvas.getContext('webgl', { premultipliedAlpha: false }) ||
    canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null
  if (!gl) return null
  try {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
    return new ColorFilterRenderer(canvas, video, gl)
  } catch (err) {
    console.warn('Color filter renderer unavailable:', err)
    return null
  }
}

function buildProgram(gl: WebGLRenderingContext): WebGLProgram {
  const vs = compile(gl, gl.VERTEX_SHADER, VERT)
  const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG)
  const program = gl.createProgram()
  if (!program) throw new Error('createProgram failed')
  gl.attachShader(program, vs)
  gl.attachShader(program, fs)
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link failed: ${gl.getProgramInfoLog(program)}`)
  }
  return program
}

function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type)
  if (!shader) throw new Error('createShader failed')
  gl.shaderSource(shader, src)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`Shader compile failed: ${gl.getShaderInfoLog(shader)}`)
  }
  return shader
}

function createTexture(gl: WebGLRenderingContext): WebGLTexture {
  const tex = gl.createTexture()
  if (!tex) throw new Error('createTexture failed')
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  return tex
}
