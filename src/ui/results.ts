import type { GameStore } from '../game/store.ts'
import type { TugStep } from '../game/types.ts'
import { fetchReplay } from '../game/replay.ts'
import { MAX_HP } from '../game/scoring.ts'

/**
 * End-of-match results screen — where the suspense pays off: the opponent, the
 * tug-of-war HP, and the winner are all revealed here.
 *
 * The match is already resolved live by the store (HP hit 0). This screen just
 * recaps it: fetch the opponent replay video, then replay the recorded
 * tug-of-war onto the HP bar + breakdown, and show final scores + the winner.
 */
export interface ResultsNav {
  onHome: () => void
  onRematch: () => void
}

export class ResultsScreen {
  private root: HTMLElement
  private nav: ResultsNav
  private updateBreakdown: (() => void) | null = null

  constructor(root: HTMLElement, nav: ResultsNav) {
    this.root = root
    this.nav = nav
    this.root.classList.add('results', 'hidden')
  }

  async show(store: GameStore, matchId: string): Promise<void> {
    // Replay is a nice-to-have — never let a missing/failed replay block the
    // results screen (online matches have no mock replay and the endpoint is TBD).
    const replay = await fetchReplay(matchId).catch(() => ({ videoUrl: null, timeline: [] }))

    const s = store.getState()
    const me = s.players.me
    const opp = s.players.opponent
    const winner = s.winner ?? (me.hp >= opp.hp ? 'me' : 'opponent')
    const youWon = winner === 'me'
    const steps = store.tugHistory
    const finalHp = { me: me.hp, opponent: opp.hp }

    this.root.classList.remove('hidden', 'win', 'lose')
    this.root.classList.add(youWon ? 'win' : 'lose')

    this.root.innerHTML = `
      <div class="results-deco"></div>
      <div class="results-inner">
        <div class="results-header">
          <div class="results-chip">🏆 Match complete</div>
          <div class="results-banner">${youWon ? 'You won!' : 'So close!'}</div>
          <div class="results-tagline">
            ${youWon ? `You out-signed ${opp.name}` : `${opp.name} edged you out`}
          </div>
        </div>

        <div class="results-tug">
          <div class="tug-names">
            <span class="me">${me.name}</span>
            <span class="opp">${opp.name}</span>
          </div>
          <div class="tug-bar">
            <div class="tug-fill" data-tug></div>
            <div class="tug-divider" data-divider></div>
          </div>
        </div>

        <div class="results-scores">
          <div class="score-card ${youWon ? 'winner' : ''}">
            <div class="score-name">${me.name}</div>
            <div class="score-num" data-myscore>0</div>
            <div class="score-sub">${s.cleared} signs · ${Math.round(finalHp.me)} HP</div>
          </div>
          <div class="score-vs">VS</div>
          <div class="score-card ${!youWon ? 'winner' : ''}">
            <div class="score-name">${opp.name}</div>
            <div class="score-num" data-oppscore>0</div>
            <div class="score-sub">${store.opponentCleared()} signs · ${Math.round(finalHp.opponent)} HP</div>
          </div>
        </div>

        <div class="results-section-title">Opponent replay</div>
        <div class="replay-frame" data-replay></div>

        <div class="results-section-title">Tug-of-war log</div>
        <div class="breakdown-wrap" data-bdwrap>
          <ul class="breakdown-list" data-breakdown></ul>
          <div class="breakdown-cue" aria-hidden="true">⌄</div>
        </div>
      </div>

      <div class="results-actions">
        <button class="btn-tactile btn-light" type="button" data-home>Home</button>
        <button class="btn-tactile btn-mint" type="button" data-again>Play again</button>
      </div>
    `

    this.root.querySelector('[data-home]')?.addEventListener('click', () => {
      this.hide()
      this.nav.onHome()
    })
    this.root.querySelector('[data-again]')?.addEventListener('click', () => {
      this.hide()
      this.nav.onRematch()
    })

    this.mountReplay(replay.videoUrl)
    this.setupBreakdownScroll()
    void this.animate(steps, me.score, opp.score, finalHp)
  }

  hide(): void {
    this.root.classList.add('hidden')
  }

  /**
   * Make the tug-of-war log's scrollability visible: edge fades + a bobbing cue
   * that appear only when there's hidden content above/below. (The scrollbar
   * itself is also styled slim rather than hidden.)
   */
  private setupBreakdownScroll(): void {
    const wrap = this.root.querySelector<HTMLDivElement>('[data-bdwrap]')
    const list = this.root.querySelector<HTMLUListElement>('[data-breakdown]')
    if (!wrap || !list) return
    const update = (): void => {
      const overflowing = list.scrollHeight > list.clientHeight + 2
      const atTop = list.scrollTop <= 2
      const atBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 2
      wrap.classList.toggle('show-top', overflowing && !atTop)
      wrap.classList.toggle('show-bottom', overflowing && !atBottom)
    }
    list.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    this.updateBreakdown = update
    update()
  }

  private mountReplay(videoUrl: string | null): void {
    const frame = this.root.querySelector<HTMLDivElement>('[data-replay]')
    if (!frame) return
    if (videoUrl) {
      const video = document.createElement('video')
      video.src = videoUrl
      video.autoplay = true
      video.loop = true
      video.muted = true
      video.playsInline = true
      video.className = 'replay-video'
      frame.append(video)
    } else {
      frame.innerHTML = `<div class="replay-loading">Loading replay…<span class="replay-hint">(served by the game server once connected)</span></div>`
    }
  }

  private async animate(
    steps: TugStep[],
    myScore: number,
    oppScore: number,
    finalHp: { me: number; opponent: number },
  ): Promise<void> {
    const fill = this.root.querySelector<HTMLDivElement>('[data-tug]')
    const divider = this.root.querySelector<HTMLDivElement>('[data-divider]')
    const list = this.root.querySelector<HTMLUListElement>('[data-breakdown]')

    const setTug = (myHp: number, oppHp: number): void => {
      const pct = (myHp / (myHp + oppHp || 1)) * 100
      if (fill) fill.style.width = `${pct}%`
      if (divider) divider.style.left = `${pct}%`
    }
    setTug(MAX_HP, MAX_HP)

    this.countUp('[data-myscore]', myScore)
    this.countUp('[data-oppscore]', oppScore)

    for (const step of steps) {
      setTug(step.myHp, step.oppHp)
      if (list) {
        const who = step.winner === 'me' ? 'me' : step.winner === 'opponent' ? 'opponent' : 'tie'
        const li = document.createElement('li')
        li.className = `breakdown-row ${who}`
        const sign = step.delta > 0 ? '+' : ''
        li.innerHTML = `
          <span class="bd-left">
            <span class="bd-badge">${who === 'me' ? '✓' : who === 'opponent' ? '✕' : '–'}</span>
            <span>
              <span class="bd-word">${step.word.text}</span>
              <span class="bd-who">${who === 'me' ? 'You scored' : who === 'opponent' ? 'Rival scored' : 'No hit'}</span>
            </span>
          </span>
          <span class="bd-delta">${step.delta === 0 ? '–' : sign + step.delta}</span>`
        list.append(li)
        list.scrollTop = list.scrollHeight
        this.updateBreakdown?.()
      }
      await delay(steps.length > 20 ? 60 : 150)
    }
    // Land on the true final HP — covers online matches (no tug history) and any
    // rounding from the step-by-step replay.
    setTug(finalHp.me, finalHp.opponent)
    this.updateBreakdown?.()
  }

  private countUp(selector: string, target: number): void {
    const el = this.root.querySelector<HTMLElement>(selector)
    if (!el) return
    const steps = 30
    let i = 0
    const tick = (): void => {
      i += 1
      el.textContent = String(Math.round((target * i) / steps))
      if (i < steps) requestAnimationFrame(tick)
    }
    tick()
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
