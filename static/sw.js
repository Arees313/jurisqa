const CACHE_NAME = 'juris-qa-v1';
const urlsToCache = [
  '/',
  '/static/index.html',
  '/static/manifest.json',
  '/static/flag.png',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// Install event - cache resources
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
      .catch(err => {
        console.log('Cache install failed:', err);
      })
  );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        if (response) {
          return response;
        }
        
        // For API calls, try network first
        if (event.request.url.includes('/ask')) {
          return fetch(event.request).catch(() => {
            // Return offline message for API calls
            return new Response(
              JSON.stringify({
                answer: "🔌 You are currently offline. Please check your internet connection and try again.",
                source: "Offline Mode"
              }),
              {
                headers: { 'Content-Type': 'application/json' },
                status: 200
              }
            );
          });
        }
        
        return fetch(event.request);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
