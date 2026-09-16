# Vendored assets

## `iconpark.json` — the IconPark Outline icon collection

Every icon in the UI is an inline `<svg>` body from Iconify's **IconPark
Outline** set. `iconpark.json` is that *whole* collection, vendored: the app
never touches the network for an icon, and it never has to fetch glyphs one at a
time.

| | |
|---|---|
| Source | `@iconify-json/icon-park-outline` **v1.2.4**, via jsDelivr: `https://cdn.jsdelivr.net/npm/@iconify-json/icon-park-outline@1.2.4/icons.json` |
| Merged in | the same package's `info.json`, stored as the collection's `info` block, so the file names its own author and licence |
| Icons | 2658 (icon set version 1.4.2 — `info.version`) |
| Author | ByteDance — <https://github.com/bytedance/IconPark> |
| Licence | Apache-2.0 — <https://github.com/bytedance/IconPark/blob/master/LICENSE> |
| Vendored | 2026-09-15, sha256 `c49ab88135f10d9382b6289c1e650b0e8f148af9a9a77b5ade733145b2eff845` |
| Origin of the geometry | 48×48 viewBox, `fill="none"`, `stroke="currentColor"`, `stroke-width="4"` — which is why icons inherit the surrounding text colour |

**Why a vendored package and not the Iconify API?** `https://api.iconify.design/icon-park-outline.json`
will *not* return a collection: asked for the whole set it answers a bare `404`
(it serves only the names you list). That is why this file began life as a
hand-picked subset of ~22 glyphs, re-fetched one icon at a time whenever the UI
needed a new one. The published `@iconify-json/*` packages are the entire set, so
vendoring one of those ends the per-icon downloads.

### Using an icon

Markup: `<i data-i="speed-one"></i>` — `hydrateIcons()` replaces it with the
`<svg>` at boot. Or from script: `setIcon(el, "speed-one")`.

**A name that is not in here renders as nothing.** An icon-only button then
becomes an empty circle rather than a visibly wrong glyph, which is how the
fullscreen button (named `expand`, never vendored) and later the speed button
(before `speed-one` was added) both broke in the same silent way. To check a
name:

```bash
python -c "import json;print('speed-one' in json.load(open('web/vendor/iconpark.json'))['icons'])"
```

### Refreshing or adding icons

1. Pick names from <https://icon-sets.iconify.design/icon-park-outline/>.
2. Re-fetch the package's `icons.json` (and `info.json`), keeping the file's
   `prefix` / `width` / `height` / `icons` shape; the loader reads `icons[name].body`
   and hands the whole collection to `Iconify.addCollection()` when that runtime
   is present, so anything else the collection carries is preserved.
3. Update the table above (version, count, sha256) — re-run the vendor step and
   copy the fresh digest rather than editing by hand.
4. **Bump the `?v=` in `fetch("vendor/iconpark.json?v=N")` in `web/index.html`.**
   The browser caches this file hard, and a stale copy makes a newly added name
   hydrate to nothing — the same silent failure as above, but now with a correct
   vendor file on disk to mislead you.
