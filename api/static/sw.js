// Qwen3-TTS Studio service worker — enables PWA install.
// Strategy: navigations network-first (fresh UI after deploys),
// /static/* cache-first, API/MCP/health always network (never cached).
const CACHE = 'tts-studio-v1';
const PRECACHE = [
  '/',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/icon-maskable-512.png',
  '/static/apple-touch-icon.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;
  // never touch API, MCP, health, or clip/profile audio endpoints
  if (url.pathname.startsWith('/v1') || url.pathname === '/mcp' ||
      url.pathname === '/health' || url.pathname.startsWith('/mcp/')) return;

  if (req.mode === 'navigate') {
    e.respondWith(
      fetch(req).then(resp => {
        const cp = resp.clone();
        caches.open(CACHE).then(c => c.put('/', cp));
        return resp;
      }).catch(() => caches.match('/'))
    );
    return;
  }

  if (url.pathname.startsWith('/static/')) {
    e.respondWith(
      caches.match(req).then(hit => hit || fetch(req).then(resp => {
        if (resp.ok) {
          const cp = resp.clone();
          caches.open(CACHE).then(c => c.put(req, cp));
        }
        return resp;
      }))
    );
  }
});
