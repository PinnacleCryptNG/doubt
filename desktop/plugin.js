import { TRANSCRIPT_DIRECTIVE_AREA } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const DOUBT = 'oklch(0.63 0.21 25)'
const pct = x => (x == null ? 'n/a' : `${Math.round(100 * x)}%`)

function useRest(ctx, path, everyMs) {
  const [data, setData] = useState(null)
  useEffect(() => {
    let live = true
    const load = () => ctx.rest(path).then(d => live && setData(d), () => {})
    load()
    const t = everyMs ? setInterval(load, everyMs) : null
    return () => { live = false; t && clearInterval(t) }
  }, [path])
  return data
}

function wordStyle(conf) {
  const doubt = Math.max(0, Math.min(1, (0.9 - conf) / 0.9))
  if (doubt <= 0) return undefined
  return {
    backgroundColor: `color-mix(in oklch, ${DOUBT} ${Math.round(doubt * 38)}%, transparent)`,
    textDecorationLine: doubt > 0.45 ? 'underline' : 'none',
    textDecorationStyle: 'wavy',
    textDecorationColor: DOUBT,
    textUnderlineOffset: '3px',
    borderRadius: '2px'
  }
}

function Heat({ text, words }) {
  const parts = []
  let pos = 0
  words.forEach((w, i) => {
    if (w.start > pos) parts.push(text.slice(pos, w.start))
    const alt = w.alts && w.alts[0] ? ` · next guess ${JSON.stringify(w.alts[0][0].trim())} ${pct(w.alts[0][1])}` : ''
    parts.push(jsx('span', { style: wordStyle(w.conf), title: `${pct(w.conf)} sure${alt}`, children: w.text }, i))
    pos = w.start + w.text.length
  })
  parts.push(text.slice(pos))
  return jsx('div', { className: 'whitespace-pre-wrap leading-relaxed', children: parts })
}

function Meter({ value }) {
  return jsx('div', {
    className: 'h-1 w-full overflow-hidden rounded-full bg-(--ui-stroke-secondary)',
    children: jsx('div', { style: { width: pct(value ?? 0), height: '100%', background: `color-mix(in oklch, ${DOUBT}, var(--ui-accent) ${Math.round(100 * (value ?? 0))}%)` } })
  })
}

function ReplyCard({ rec }) {
  if (!rec.native) {
    return jsx('div', { className: 'text-(--ui-text-tertiary)', children: `${rec.model} returned no certainty scores for this reply.` })
  }
  const s = rec.summary
  return jsxs('div', {
    className: 'flex flex-col gap-2',
    children: [
      jsx(Heat, { text: rec.text, words: rec.words }),
      jsx(Meter, { value: s.confidence }),
      jsx('div', { className: 'text-xs text-(--ui-text-tertiary)', children: `${pct(s.confidence)} sure · ${s.low_words} shaky words · ${rec.model}` })
    ]
  })
}

function GuardRow({ rec }) {
  const verdict = { pass: 'passed', asked: 'asked you', blocked: 'blocked', warned: 'logged' }[rec.decision] || rec.decision
  return jsxs('div', {
    className: 'flex flex-col gap-1 border-t border-(--ui-stroke-secondary) pt-2',
    children: [
      jsx('code', { className: 'text-xs', children: jsx(Heat, { text: rec.actual, words: rec.words }) }),
      jsx('div', { className: 'text-xs text-(--ui-text-tertiary)', children: `${verdict} · ${pct(rec.confidence)} sure · re-derived ${pct(rec.agreement)} identically` })
    ]
  })
}

function Pane({ ctx }) {
  const replies = useRest(ctx, '/records?kind=reply&limit=1', 3000)
  const guards = useRest(ctx, '/records?kind=guard&limit=6', 3000)
  const last = replies && replies.records[0]
  return jsxs('div', {
    className: 'flex h-full flex-col gap-4 overflow-y-auto p-3 text-sm',
    children: [
      jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('div', { className: 'font-medium', children: 'Last reply' }),
        last ? jsx(ReplyCard, { rec: last }) : jsx('div', { className: 'text-(--ui-text-tertiary)', children: 'Nothing scored yet. Doubt needs model.streaming: false.' })
      ] }),
      jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('div', { className: 'font-medium', children: 'Commands checked' }),
        ...((guards && guards.records) || []).map(r => jsx(GuardRow, { rec: r }, r.id))
      ] })
    ]
  })
}

function DirectiveCard({ ctx, id }) {
  const rec = useRest(ctx, `/records/${encodeURIComponent(id)}`, 0)
  if (!rec) return null
  return jsx('div', {
    className: 'rounded-md border border-(--ui-stroke-secondary) p-3 text-sm',
    children: rec.kind === 'guard' ? jsx(GuardRow, { rec }) : jsx(ReplyCard, { rec })
  })
}

export default {
  id: 'doubt',
  name: 'Doubt',
  register(ctx) {
    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'doubt',
      data: { placement: 'right', width: '320px' },
      render: () => jsx(Pane, { ctx })
    })
    ctx.register({
      id: 'card',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      data: { name: 'doubt', render: ({ attrs }) => jsx(DirectiveCard, { ctx, id: attrs.id }) }
    })
  }
}
