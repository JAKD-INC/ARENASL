/**
 * Guided coach overlay — a short spotlight tour that teaches the controls and
 * the core loop the first time the player enters Practice (and replayable from
 * the "?" button). Each step optionally spotlights a real on-screen element by
 * selector; steps with no target show a centered card over a dimmed screen.
 *
 * Purely presentational: it knows nothing about the game, so it can point at any
 * element. The caller supplies the steps and decides when to run it.
 */

export interface CoachStep {
  /** CSS selector of the element to spotlight; omit for a centered card. */
  target?: string
  title: string
  body: string
}

export class CoachOverlay {
  private root: HTMLElement
  private hole: HTMLDivElement
  private card: HTMLDivElement
  private titleEl: HTMLDivElement
  private bodyEl: HTMLDivElement
  private dotsEl: HTMLDivElement
  private nextBtn: HTMLButtonElement
  private steps: CoachStep[] = []
  private i = 0
  private onDone?: () => void
  private reposition = (): void => this.position()

  constructor(root: HTMLElement) {
    this.root = root
    root.classList.add('coach', 'hidden')
    root.innerHTML = `
      <div class="coach-hole" data-hole></div>
      <div class="coach-card" data-card>
        <div class="coach-title" data-title></div>
        <div class="coach-body" data-body></div>
        <div class="coach-foot">
          <div class="coach-dots" data-dots></div>
          <div class="coach-actions">
            <button class="coach-skip" type="button" data-skip>Skip</button>
            <button class="coach-next btn-tactile btn-mint" type="button" data-next>Next</button>
          </div>
        </div>
      </div>
    `
    const q = <T extends HTMLElement>(sel: string): T => root.querySelector<T>(sel)!
    this.hole = q('[data-hole]')
    this.card = q('[data-card]')
    this.titleEl = q('[data-title]')
    this.bodyEl = q('[data-body]')
    this.dotsEl = q('[data-dots]')
    this.nextBtn = q('[data-next]')
    q<HTMLButtonElement>('[data-next]').addEventListener('click', () => this.advance())
    q<HTMLButtonElement>('[data-skip]').addEventListener('click', () => this.finish())
  }

  /** Run the tour. `onDone` fires when finished or skipped. */
  start(steps: CoachStep[], onDone?: () => void): void {
    if (!steps.length) return
    this.steps = steps
    this.i = 0
    this.onDone = onDone
    this.root.classList.remove('hidden')
    window.addEventListener('resize', this.reposition)
    this.render()
  }

  private advance(): void {
    if (this.i >= this.steps.length - 1) this.finish()
    else {
      this.i += 1
      this.render()
    }
  }

  private finish(): void {
    this.root.classList.add('hidden')
    window.removeEventListener('resize', this.reposition)
    this.onDone?.()
  }

  private render(): void {
    const step = this.steps[this.i]
    this.titleEl.textContent = step.title
    this.bodyEl.textContent = step.body
    this.nextBtn.textContent = this.i === this.steps.length - 1 ? 'Got it' : 'Next'
    this.dotsEl.innerHTML = this.steps
      .map((_, k) => `<span class="coach-dot${k === this.i ? ' on' : ''}"></span>`)
      .join('')
    // Let the card lay out before measuring it.
    requestAnimationFrame(() => this.position())
  }

  private position(): void {
    const step = this.steps[this.i]
    const el = step?.target ? document.querySelector(step.target) : null
    const rect = el?.getBoundingClientRect()
    const card = this.card

    if (rect && rect.width > 0 && rect.height > 0) {
      const pad = 10
      this.hole.style.opacity = '1'
      this.hole.style.left = `${rect.left - pad}px`
      this.hole.style.top = `${rect.top - pad}px`
      this.hole.style.width = `${rect.width + pad * 2}px`
      this.hole.style.height = `${rect.height + pad * 2}px`

      const cardH = card.offsetHeight || 170
      const placeBelow = rect.bottom + 14 + cardH < window.innerHeight
      card.style.left = '50%'
      card.style.transform = 'translateX(-50%)'
      if (placeBelow) {
        card.style.top = `${rect.bottom + 14}px`
        card.style.bottom = 'auto'
      } else {
        card.style.top = 'auto'
        card.style.bottom = `${window.innerHeight - rect.top + 14}px`
      }
    } else {
      // No target → dim everything (zero-size hole) and center the card.
      this.hole.style.opacity = '1'
      this.hole.style.left = '50%'
      this.hole.style.top = '50%'
      this.hole.style.width = '0px'
      this.hole.style.height = '0px'
      card.style.left = '50%'
      card.style.top = '50%'
      card.style.bottom = 'auto'
      card.style.transform = 'translate(-50%, -50%)'
    }
  }
}
