// One frame clock for the moveset review: the source preview, the GVHMR overlay and the 3D view.
//
// Clip frames are 1:1 across all three. While playing, the first video that can play is the master
// clock (its decoder paces playback) and the others are pulled back when they drift by more than two
// frames; with no video at all a wall clock drives the 3D view. When paused, every view sits on the
// middle of the current frame so stepping is exact.

export class Timeline {
  constructor() {
    this.fps = 25;
    this.n = 1;
    this.contact = 0;
    this.t = 0;
    this.playing = false;
    this.rate = 1;
    this.loop = true;
    this.videos = [];
    this.listeners = new Set();
    this._raf = 0;
    this._clock0 = 0;
    this._t0 = 0;
  }

  get duration() { return this.n / this.fps; }
  get frame() { return clamp(Math.floor(this.t * this.fps + 1e-4), 0, this.n - 1); }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit() { for (const fn of this.listeners) fn(this); }

  /** videos: [{ el: HTMLVideoElement, start: seconds of the clip's first frame on the media timeline }] */
  load({ fps, n, contact, videos }) {
    this.pause(false);
    this.fps = fps || 25;
    this.n = Math.max(1, n | 0);
    this.contact = clamp(contact ?? 0, 0, this.n - 1);
    this.videos = videos.map(({ el, start }) => {
      const v = { el, start: start || 0, ok: true };
      el.onerror = () => { v.ok = false; };
      el.playbackRate = this.rate;
      return v;
    });
    this.seekFrame(0);
  }

  master() {
    return this.videos.find((v) => v.ok && v.el.readyState >= 2 && !v.el.seeking) || null;
  }

  play() {
    if (this.playing) return;
    if (this.frame >= this.n - 1) this.t = 0;
    this.playing = true;
    this._restartClock();
    for (const v of this.videos) {
      if (!v.ok) continue;
      v.el.playbackRate = this.rate;
      this._place(v, this.t);
      v.el.play().catch(() => {});
    }
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(() => this._tick());
    this.emit();
  }

  pause(snap = true) {
    const was = this.playing;
    this.playing = false;
    cancelAnimationFrame(this._raf);
    for (const v of this.videos) v.el.pause();
    if (snap && was) this.seekFrame(this.frame);
    else this.emit();
  }

  toggle() { if (this.playing) this.pause(); else this.play(); }

  seekFrame(f) {
    f = clamp(Math.round(f), 0, this.n - 1);
    this.t = f / this.fps;
    for (const v of this.videos) if (v.ok) this._place(v, (f + 0.5) / this.fps);
    if (this.playing) this._restartClock();
    this.emit();
  }

  step(delta) {
    if (this.playing) this.pause(false);
    this.seekFrame(this.frame + delta);
  }

  setRate(rate) {
    this.rate = rate;
    for (const v of this.videos) v.el.playbackRate = rate;
    this._restartClock();
    this.emit();
  }

  setLoop(on) { this.loop = on; this.emit(); }

  setMuted(muted) { for (const v of this.videos) v.el.muted = muted; }

  _restartClock() {
    this._clock0 = performance.now();
    this._t0 = this.t;
  }

  _place(v, t) {
    const end = Number.isFinite(v.el.duration) ? v.el.duration - 0.001 : Infinity;
    const target = Math.max(0, Math.min(v.start + t, end));
    if (Math.abs(v.el.currentTime - target) > 0.0005) v.el.currentTime = target;
  }

  _tick() {
    if (!this.playing) return;
    const m = this.master();
    let t;
    if (m && !m.el.paused) {
      t = m.el.currentTime - m.start;
      // keep the wall clock on the master: if every video drops out (seeking, starved), time carries on from
      // here instead of jumping to wherever the clock has run since play() and wrapping the loop early
      this._t0 = t;
      this._clock0 = performance.now();
    } else {
      t = this._t0 + ((performance.now() - this._clock0) / 1000) * this.rate;
    }
    const ended = (m && m.el.ended) || t >= this.duration - 0.5 / this.fps;
    if (ended) {
      if (!this.loop) {
        this.playing = false;
        for (const v of this.videos) v.el.pause();
        this.seekFrame(this.n - 1);
        return;
      }
      this.t = 0;
      this._restartClock();
      for (const v of this.videos) {
        if (!v.ok) continue;
        this._place(v, 0);
        v.el.play().catch(() => {});
      }
    } else {
      this.t = Math.max(0, t);
      for (const v of this.videos) {
        if (!v.ok || v === m) continue;
        if (Math.abs(v.el.currentTime - v.start - this.t) > 2 / this.fps) this._place(v, this.t);
        if (v.el.paused && !v.el.ended) v.el.play().catch(() => {});
      }
    }
    this.emit();
    this._raf = requestAnimationFrame(() => this._tick());
  }
}

function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
