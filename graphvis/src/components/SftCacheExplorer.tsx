import React, { useMemo, useState } from 'react';
import {
  useSftCaches,
  useSftCacheItems,
  useSftCacheItemDetail,
} from '../hooks/useSftCaches';
import './GraphCanvas.css';

type VerifiedFilter = 'all' | 'verified' | 'rejected';

export const SftCacheExplorer: React.FC = () => {
  const { caches, loading: cachesLoading, error: cachesError, reload: reloadCaches } = useSftCaches();

  const [selectedCacheId, setSelectedCacheId] = useState<string | null>(null);
  const effectiveCacheId = selectedCacheId ?? caches[0]?.id ?? null;

  const [verifiedFilter, setVerifiedFilter] = useState<VerifiedFilter>('all');
  const [knowledgeFilter, setKnowledgeFilter] = useState<string>('');

  // Single counter shared by the items + item-detail hooks. Bumping it
  // forces both downstream queries to re-fetch, so a sidebar Refresh
  // updates the main pane too instead of leaving stale data.
  const [refreshToken, setRefreshToken] = useState(0);
  const handleRefresh = () => {
    reloadCaches();
    setRefreshToken(t => t + 1);
  };

  const filters = useMemo(
    () => ({
      verified: verifiedFilter === 'verified' ? ('true' as const) : verifiedFilter === 'rejected' ? ('false' as const) : undefined,
      knowledge_id: knowledgeFilter || undefined,
      limit: 1000,
    }),
    [verifiedFilter, knowledgeFilter],
  );

  const { items, loading: itemsLoading, error: itemsError } = useSftCacheItems(effectiveCacheId, filters, refreshToken);
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const { detail, loading: detailLoading } = useSftCacheItemDetail(effectiveCacheId, selectedItemId, refreshToken);

  return (
    <div style={{ display: 'flex', height: '100%', fontFamily: '"Inter", sans-serif', textAlign: 'left', background: '#0f111a', color: '#fff' }}>
      {/* Sidebar */}
      <div style={{ width: '380px', borderRight: '1px solid rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column', background: 'rgba(15, 17, 26, 0.5)' }}>
        <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', color: '#ccc' }}>SFT Caches</h3>
            <button
              onClick={handleRefresh}
              disabled={cachesLoading}
              style={{ background: 'transparent', border: '1px solid #444', color: cachesLoading ? '#555' : '#888', borderRadius: '4px', padding: '4px 8px', fontSize: '0.75rem', cursor: cachesLoading ? 'default' : 'pointer' }}
            >
              {cachesLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
          <div style={{ fontSize: '11px', color: '#666' }}>
            cache_id ({caches.length})
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {cachesError && <div style={{ padding: '20px', color: '#e74c3c', fontSize: '12px' }}>Failed to load: {cachesError}</div>}
          {!cachesError && caches.length === 0 && !cachesLoading && (
            <div style={{ padding: '40px 20px', textAlign: 'center', color: '#555', fontSize: '13px' }}>No SFT caches yet.</div>
          )}
          {caches.map(c => {
            const selected = effectiveCacheId === c.id;
            const verifiedRate = c.total_items > 0 ? c.verified_items / c.total_items : 0;
            const color = c.total_items === 0
              ? '#666'
              : c.verified_items === 0
              ? '#ff4d4d'
              : verifiedRate === 1
              ? '#2ecc71'
              : '#f1c40f';
            return (
              <div
                key={c.id}
                onClick={() => { setSelectedCacheId(c.id); setSelectedItemId(null); setVerifiedFilter('all'); setKnowledgeFilter(''); }}
                style={{
                  padding: '12px 20px',
                  cursor: 'pointer',
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                  background: selected ? 'rgba(46, 204, 113, 0.08)' : 'transparent',
                  borderLeft: selected ? '3px solid #2ecc71' : '3px solid transparent',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: '11px', color, fontWeight: 600 }}>
                    {c.verified_items}/{c.total_items}
                  </span>
                  {c.unique_knowledges > 0 && (
                    <span style={{ fontSize: '10px', color: '#888' }}>· {c.unique_knowledges} knowledges</span>
                  )}
                  {c.latest_item_at && (
                    <span style={{ fontSize: '10px', color: '#666', marginLeft: 'auto' }}>
                      {new Date(c.latest_item_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '12px', color: selected ? '#fff' : '#bbb', lineHeight: 1.4, fontFamily: 'monospace', wordBreak: 'break-all', marginBottom: '2px' }}>
                  {c.id}
                </div>
                {(c.library_name || c.granular_id) && (
                  <div style={{ fontSize: '10px', color: '#666' }}>
                    {c.library_name && <span>{c.library_name}</span>}
                    {c.library_name && c.granular_id && <span> · </span>}
                    {c.granular_id && <span style={{ fontFamily: 'monospace' }}>{c.granular_id}</span>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '30px 40px', background: 'radial-gradient(circle at top right, rgba(46, 204, 113, 0.04), transparent 40%)' }}>
        <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
          {!effectiveCacheId ? (
            <div style={{ textAlign: 'center', padding: '120px 20px', color: '#555' }}>
              <div style={{ fontSize: '40px', marginBottom: '12px' }}>📚</div>
              <div style={{ fontSize: '14px' }}>Select a cache on the left to inspect its items.</div>
            </div>
          ) : (
            <>
              <div style={{ marginBottom: '6px', fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px' }}>Cache</div>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 600, margin: 0, marginBottom: '8px', color: '#eee', fontFamily: 'monospace', wordBreak: 'break-all' }}>
                {effectiveCacheId}
              </h2>
              {(() => {
                const c = caches.find(x => x.id === effectiveCacheId);
                if (!c?.description) return null;
                return <p style={{ fontSize: '12px', color: '#888', marginTop: 0, marginBottom: '20px' }}>{c.description}</p>;
              })()}

              {/* Filters */}
              <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
                <select value={verifiedFilter} onChange={e => setVerifiedFilter(e.target.value as VerifiedFilter)} style={selectStyle}>
                  <option value="all">All</option>
                  <option value="verified">Verified only</option>
                  <option value="rejected">Rejected only</option>
                </select>
                <input
                  type="text"
                  placeholder="knowledge_id filter..."
                  value={knowledgeFilter}
                  onChange={e => setKnowledgeFilter(e.target.value)}
                  style={{ ...selectStyle, flex: 2, minWidth: '180px' }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Items ({items.length})
                </div>
              </div>
              <div style={{ maxHeight: '420px', overflowY: 'auto', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', padding: '12px', background: 'rgba(0,0,0,0.15)' }}>
                {itemsError && (
                  <div style={{ color: '#e74c3c', fontSize: '12px' }}>Failed to load items: {itemsError}</div>
                )}
                {itemsLoading && (
                  <div style={{ color: '#888', fontSize: '12px', fontStyle: 'italic' }}>Loading items...</div>
                )}
                {!itemsLoading && items.length === 0 && !itemsError && (
                  <div style={{ padding: '40px', textAlign: 'center', color: '#666', border: '1px dashed #333', borderRadius: '8px' }}>
                    No items match these filters.
                  </div>
                )}
                {items.map(it => {
                  const selected = selectedItemId === it.id;
                  return (
                    <div
                      key={it.id}
                      onClick={() => setSelectedItemId(it.id)}
                      style={{
                        marginBottom: '10px',
                        padding: '10px 14px',
                        cursor: 'pointer',
                        background: selected
                          ? 'rgba(46, 204, 113, 0.1)'
                          : it.verified ? 'rgba(46, 204, 113, 0.04)' : 'rgba(255, 77, 77, 0.04)',
                        border: `1px solid ${selected ? '#2ecc71' : 'rgba(255,255,255,0.06)'}`,
                        borderRadius: '6px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
                        <span style={{ fontSize: '11px', color: it.verified ? '#2ecc71' : '#ff4d4d', fontWeight: 600 }}>
                          {it.verified ? '✓ verified' : '✗ rejected'}
                        </span>
                        {it.conclusion && (
                          <span style={{ fontSize: '10px', color: '#888' }}>{it.conclusion}</span>
                        )}
                        <span style={{ fontSize: '10px', color: '#888', fontFamily: 'monospace' }}>{it.knowledge_title}</span>
                        <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#666' }}>#{it.id}</span>
                      </div>
                      <div style={{ fontSize: '12px', color: '#bbb', lineHeight: 1.5, display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                        {extractTaskInstruction(it.question)}
                      </div>
                    </div>
                  );
                })}
              </div>

              {selectedItemId !== null && (
                <ItemDetailPanel
                  loading={detailLoading}
                  detail={detail}
                  onClose={() => setSelectedItemId(null)}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// study2 stores the full `<Investigate ...>` prompt template in `question`.
// In the items list we only want the `<InvestigationTarget>` body (the actual
// task instruction); fall back to the raw question for other cache formats.
const extractTaskInstruction = (q: string): string => {
  const m = q.match(/<InvestigationTarget>\s*([\s\S]*?)\s*<\/InvestigationTarget>/);
  return m ? m[1].trim() : q;
};

const selectStyle: React.CSSProperties = {
  padding: '6px 10px',
  background: '#1a1d27',
  color: '#fff',
  border: '1px solid #333',
  borderRadius: '6px',
  outline: 'none',
  fontSize: '12px',
};

const ItemDetailPanel: React.FC<{
  loading: boolean;
  detail: import('../types').SftCacheItemDetail | null;
  onClose: () => void;
}> = ({ loading, detail, onClose }) => (
  <div style={{ marginTop: '24px', borderTop: '2px solid rgba(46, 204, 113, 0.3)', paddingTop: '20px' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
      <h3 style={{ margin: 0, fontSize: '1rem', color: '#2ecc71' }}>Item detail</h3>
      <button onClick={onClose} style={{ background: 'transparent', border: '1px solid #444', color: '#888', borderRadius: '4px', padding: '4px 10px', fontSize: '11px', cursor: 'pointer' }}>
        Close
      </button>
    </div>
    {loading && <div style={{ color: '#888', fontSize: '12px', fontStyle: 'italic' }}>Loading...</div>}
    {!loading && !detail && <div style={{ color: '#888', fontSize: '12px' }}>No detail loaded.</div>}
    {detail && (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ fontSize: '11px', color: '#888' }}>
          <span>{detail.verified ? '✓ verified' : '✗ rejected'}</span>
          <span> · {detail.conclusion || 'no conclusion'}</span>
          {detail.reward !== null && detail.reward !== undefined && (
            <span> · reward {detail.reward.toFixed(2)}</span>
          )}
          <span> · {detail.knowledge_title}</span>
          {detail.trials && (
            <span> · {detail.trials.length} turns</span>
          )}
        </div>
        <DetailBlock title="Question" body={detail.question} />
        {detail.reasoning && <DetailBlock title="Reasoning" body={detail.reasoning} />}
        <DetailBlock title="Answer (submitted code)" body={detail.answer} />
        {detail.verifier_reasoning && (
          <DetailBlock title="Verifier reasoning" body={detail.verifier_reasoning} />
        )}
        {detail.trials && detail.trials.length > 0 && (
          <SynthesisTrajectory trials={detail.trials} verified={detail.verified} verifierReasoning={detail.verifier_reasoning} />
        )}
      </div>
    )}
  </div>
);

const DetailBlock: React.FC<{ title: string; body: string }> = ({ title, body }) => (
  <div className="trajectory-card assistant" style={{ padding: '12px 14px', borderRadius: '8px' }}>
    <div style={{ fontSize: '10px', opacity: 0.6, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{title}</div>
    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '12px', color: '#ddd', fontFamily: '"SF Mono", "Menlo", monospace' }}>
      {body || <span style={{ color: '#555', fontStyle: 'italic' }}>(empty)</span>}
    </pre>
  </div>
);

// --- Trajectory rendering ---
// The synthesis solver speaks the XML-tag protocol (write_and_run / submit /
// search_library_doc), not OpenAI tool_calls. This component renders each
// turn with a role-colored card and pulls out tag content for readability.

const flattenContent = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((p: any) =>
        p?.type === 'thinking' ? `<think>${p.thinking ?? ''}</think>` : (p?.text ?? '')
      )
      .join('');
  }
  return content == null ? '' : String(content);
};

const SynthesisTrajectory: React.FC<{
  trials: import('../types').SftTrialMessage[];
  verified: boolean;
  verifierReasoning: string;
}> = ({ trials, verified, verifierReasoning }) => {
  return (
    <div style={{ borderTop: '1px solid #333', paddingTop: '14px', marginTop: '4px' }}>
      <div style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '10px' }}>
        Investigation trajectory ({trials.length} turns)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {trials.map((t, i) => (
          <TrialTurn key={i} turn={t} index={i} />
        ))}
        <div
          style={{
            padding: '12px 14px',
            borderRadius: '8px',
            border: `1px solid ${verified ? 'rgba(46, 204, 113, 0.4)' : 'rgba(255, 77, 77, 0.4)'}`,
            background: verified ? 'rgba(46, 204, 113, 0.06)' : 'rgba(255, 77, 77, 0.06)',
          }}
        >
          <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.5px', color: verified ? '#2ecc71' : '#ff4d4d', marginBottom: '6px', fontWeight: 600 }}>
            {verified ? '✓ Verifier accepted' : '✗ Verifier rejected'}
          </div>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '12px', color: '#ddd', fontFamily: '"SF Mono", "Menlo", monospace' }}>
            {verifierReasoning || <span style={{ color: '#555', fontStyle: 'italic' }}>(no reasoning)</span>}
          </pre>
        </div>
      </div>
    </div>
  );
};

const ROLE_STYLES: Record<string, { color: string; bg: string; border: string; label: string }> = {
  system: { color: '#9b59b6', bg: 'rgba(155, 89, 182, 0.06)', border: 'rgba(155, 89, 182, 0.25)', label: 'System' },
  user: { color: '#61dafb', bg: 'rgba(97, 218, 251, 0.06)', border: 'rgba(97, 218, 251, 0.25)', label: 'User / Tool result' },
  assistant: { color: '#f1c40f', bg: 'rgba(241, 196, 15, 0.06)', border: 'rgba(241, 196, 15, 0.25)', label: 'Assistant' },
  tool: { color: '#2ecc71', bg: 'rgba(46, 204, 113, 0.06)', border: 'rgba(46, 204, 113, 0.25)', label: 'Tool' },
};

const TrialTurn: React.FC<{ turn: import('../types').SftTrialMessage; index: number }> = ({ turn, index }) => {
  const roleKey = String(turn.role || 'user');
  const style = ROLE_STYLES[roleKey] ?? { color: '#888', bg: 'rgba(255,255,255,0.03)', border: 'rgba(255,255,255,0.1)', label: roleKey };
  const text = flattenContent(turn.content);
  const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/);
  const visible = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

  // Extract notable XML-tag actions for quick scanning.
  const submit = visible.match(/<submit>([\s\S]*?)<\/submit>/)?.[1];
  const writeRun = visible.match(/<write_and_run>([\s\S]*?)<\/write_and_run>/)?.[1];
  const searchDoc = visible.match(/<search_library_doc>([\s\S]*?)<\/search_library_doc>/)?.[1];

  return (
    <div style={{ padding: '10px 14px', borderRadius: '8px', background: style.bg, border: `1px solid ${style.border}` }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
        <span style={{ fontSize: '9px', color: style.color, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          #{index + 1} {style.label}
        </span>
        {submit !== undefined && (
          <span style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '3px', background: 'rgba(46, 204, 113, 0.2)', color: '#2ecc71' }}>submit</span>
        )}
        {writeRun !== undefined && (
          <span style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '3px', background: 'rgba(241, 196, 15, 0.2)', color: '#f1c40f' }}>write_and_run</span>
        )}
        {searchDoc !== undefined && (
          <span style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '3px', background: 'rgba(97, 218, 251, 0.2)', color: '#61dafb' }}>search_library_doc</span>
        )}
      </div>
      {thinkMatch && (
        <div style={{ marginBottom: '8px', padding: '8px 10px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '6px', borderLeft: '2px solid #888' }}>
          <div style={{ fontSize: '9px', color: '#888', textTransform: 'uppercase', marginBottom: '4px' }}>Thought</div>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '11px', color: '#aaa', fontStyle: 'italic', fontFamily: '"SF Mono", "Menlo", monospace' }}>
            {thinkMatch[1].trim()}
          </pre>
        </div>
      )}
      <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '12px', color: '#ddd', fontFamily: '"SF Mono", "Menlo", monospace' }}>
        {visible || <span style={{ color: '#555', fontStyle: 'italic' }}>(empty)</span>}
      </pre>
    </div>
  );
};
