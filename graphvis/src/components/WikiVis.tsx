import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface WikiArticle {
  title: string;
  updated_at: string;
}

interface WikiArticleContent {
  title: string;
  content: string;
  version: string;
  updated_at: string;
}

export const WikiVis: React.FC = () => {
  const [versions, setVersions] = useState<string[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<string>('');
  const [articles, setArticles] = useState<WikiArticle[]>([]);
  const [selectedArticle, setSelectedArticle] = useState<WikiArticleContent | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [search, setSearch] = useState<string>('');

  const API_BASE = 'http://localhost:8000/api';

  const fetchVersions = async (forceUpdate = false) => {
    try {
      const res = await fetch(`${API_BASE}/wiki/versions`);
      const data = await res.json();
      setVersions(data);
      if (data.length > 0) {
        if (!selectedVersion || (forceUpdate && selectedVersion !== data[0])) {
          setSelectedVersion(data[0]);
        }
      }
      return data;
    } catch (err) {
      console.error('Failed to fetch versions:', err);
      return [];
    }
  };

  const fetchArticles = async (version: string, selectFirst = true) => {
    if (!version) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/wiki/${version}/articles`);
      const data = await res.json();
      setArticles(data);
      if (selectFirst && data.length > 0) {
        fetchArticle(data[0].title, version);
      } else if (data.length === 0) {
        setSelectedArticle(null);
      }
      return data;
    } catch (err) {
      console.error('Failed to fetch articles:', err);
      return [];
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVersions();
  }, []);

  useEffect(() => {
    fetchArticles(selectedVersion);
  }, [selectedVersion]);

  const fetchArticle = (title: string, version = selectedVersion) => {
    fetch(`${API_BASE}/wiki/${version}/article/${encodeURIComponent(title)}`)
      .then(res => res.json())
      .then(data => setSelectedArticle(data))
      .catch(err => console.error('Failed to fetch article content:', err));
  };

  const handleRefresh = async () => {
    const v = await fetchVersions();
    const currentVersion = selectedVersion || v[0];
    if (currentVersion) {
      // Re-fetch articles for the current version, but don't force select first if we already have a selection
      await fetchArticles(currentVersion, !selectedArticle);
      
      // If we have a selected article, re-fetch its content to ensure it's up to date
      if (selectedArticle) {
        fetchArticle(selectedArticle.title, currentVersion);
      }
    }
  };

  const filteredArticles = articles.filter(a => 
    a.title.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ display: 'flex', height: '100%', width: '100%', backgroundColor: '#0f111a', color: '#e6edf3' }}>
      {/* Sidebar */}
      <div style={{ 
        width: '320px', 
        borderRight: '1px solid rgba(255,255,255,0.05)', 
        display: 'flex', 
        flexDirection: 'column',
        background: 'rgba(255,255,255,0.02)',
        textAlign: 'left'
      }}>
        <div style={{ padding: '20px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.8rem', color: '#8b949e' }}>Explore Wiki Version</label>
            <button 
              onClick={handleRefresh}
              style={{ 
                background: 'transparent', 
                border: 'none', 
                color: '#61dafb', 
                cursor: 'pointer', 
                fontSize: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                padding: '4px 8px',
                borderRadius: '4px',
                transition: 'background 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(97, 218, 251, 0.1)'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
            >
              <span>Refresh</span>
              <span style={{ fontSize: '1rem' }}>↻</span>
            </button>
          </div>
          <select 
            value={selectedVersion} 
            onChange={(e) => setSelectedVersion(e.target.value)}
            style={{ 
              width: '100%', 
              padding: '10px', 
              background: '#161b22', 
              color: '#fff', 
              border: '1px solid #30363d', 
              borderRadius: '6px',
              outline: 'none',
              cursor: 'pointer',
              boxSizing: 'border-box'
            }}
          >
            {versions.map(v => <option key={v} value={v}>{v}</option>)}
          </select>

          <input 
            type="text" 
            placeholder="Search articles..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ 
              width: '100%', 
              marginTop: '15px', 
              padding: '10px', 
              background: '#0d1117', 
              color: '#fff', 
              border: '1px solid #30363d', 
              borderRadius: '6px',
              fontSize: '0.9rem',
              boxSizing: 'border-box'
            }}
          />
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
          {loading ? (
            <div style={{ padding: '20px', color: '#8b949e', textAlign: 'center' }}>Loading articles...</div>
          ) : filteredArticles.length === 0 ? (
            <div style={{ padding: '20px', color: '#8b949e', textAlign: 'center' }}>No articles found</div>
          ) : (
            filteredArticles.map(article => (
              <div 
                key={article.title}
                onClick={() => fetchArticle(article.title)}
                title={article.title}
                style={{ 
                  padding: '12px 15px', 
                  marginBottom: '4px', 
                  borderRadius: '6px', 
                  cursor: 'pointer',
                  backgroundColor: selectedArticle?.title === article.title ? 'rgba(97, 218, 251, 0.1)' : 'transparent',
                  color: selectedArticle?.title === article.title ? '#61dafb' : '#c9d1d9',
                  transition: 'all 0.2s',
                  fontSize: '0.95rem',
                  fontWeight: selectedArticle?.title === article.title ? 600 : 400,
                  border: selectedArticle?.title === article.title ? '1px solid rgba(97, 218, 251, 0.3)' : '1px solid transparent',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => {
                  if (selectedArticle?.title !== article.title) {
                    e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (selectedArticle?.title !== article.title) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
              >
                {article.title}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Content Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '40px', position: 'relative', scrollBehavior: 'smooth' }}>
        {selectedArticle ? (
          <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'left' }}>
            <div style={{ marginBottom: '30px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '20px' }}>
              <h1 style={{ fontSize: '2.5rem', marginBottom: '10px', fontWeight: 700, background: 'linear-gradient(90deg, #fff, #8b949e)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                {selectedArticle.title}
              </h1>
              <div style={{ display: 'flex', gap: '20px', color: '#8b949e', fontSize: '0.85rem' }}>
                <span>Version: <b style={{ color: '#61dafb' }}>{selectedArticle.version}</b></span>
                <span>Last Updated: {new Date(selectedArticle.updated_at).toLocaleString()}</span>
              </div>
            </div>
            
            <div className="markdown-body" style={{ color: '#c9d1d9', lineHeight: '1.6', fontSize: '1.1rem' }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedArticle.content}</ReactMarkdown>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#8b949e', flexDirection: 'column' }}>
            <div style={{ fontSize: '4rem', opacity: 0.2, marginBottom: '20px' }}>📚</div>
            <div style={{ fontSize: '1.2rem' }}>Select an article to view knowledge</div>
          </div>
        )}
      </div>

      {/* Markdown Custom Styles */}
      <style>{`
        .markdown-body h1, .markdown-body h2, .markdown-body h3 { border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.3em; margin-top: 1.5em; margin-bottom: 1em; color: #fff; }
        .markdown-body p { margin-bottom: 1em; }
        .markdown-body code { background: rgba(255,255,255,0.1); padding: 0.2em 0.4em; borderRadius: 3px; font-family: monospace; font-size: 0.9em; }
        .markdown-body pre { background: #161b22; padding: 16px; borderRadius: 8px; overflow: auto; border: 1px solid #30363d; margin: 1.5em 0; }
        .markdown-body pre code { background: transparent; padding: 0; }
        .markdown-body ul, .markdown-body ol { margin-bottom: 1em; padding-left: 2em; }
        .markdown-body blockquote { border-left: 4px solid #30363d; padding-left: 1em; color: #8b949e; font-style: italic; margin: 1.5em 0; }
        .markdown-body a { color: #58a6ff; text-decoration: none; }
        .markdown-body a:hover { text-decoration: underline; }
        .markdown-body table { border-spacing: 0; border-collapse: collapse; margin-top: 0; margin-bottom: 16px; width: 100%; }
        .markdown-body table th { font-weight: 600; background-color: rgba(255,255,255,0.05); }
        .markdown-body table th, .markdown-body table td { padding: 8px 13px; border: 1px solid #30363d; }
        .markdown-body table tr { background-color: transparent; border-top: 1px solid #30363d; }
        .markdown-body table tr:nth-child(2n) { background-color: rgba(255,255,255,0.02); }
      `}</style>
    </div>
  );
};
