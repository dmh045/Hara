/* Local document navigation. No network requests or production dependencies. */
(() => {
  const nav = document.getElementById('document-toc');
  const main = document.getElementById('document-content');
  if (!nav || !main) return;
  const entries = [...nav.querySelectorAll('.toc-entry')];
  const chapterLabel = document.getElementById('reading-chapter');
  const detailLabel = document.getElementById('reading-subsection');
  const header = document.querySelector('header');
  const desktop = matchMedia('(min-width: 781px)');
  let points = [], frame = 0, activeEntry = null, activeLink = null;

  function measure() {
    const offset = Math.ceil(header.getBoundingClientRect().height);
    document.documentElement.style.setProperty('--header-height', `${offset}px`);
    points = [];
    for (const entry of entries) {
      for (const link of entry.querySelectorAll('a[data-reading-target]')) {
        const element = document.getElementById(link.dataset.readingTarget);
        if (element) points.push({entry, link, element,
          top: element.getBoundingClientRect().top + window.scrollY});
      }
    }
    points.sort((a, b) => a.top - b.top);
    schedule();
  }

  function update() {
    frame = 0;
    if (!points.length) return;
    const y = window.scrollY + header.getBoundingClientRect().height + 26;
    let lo = 0, hi = points.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (points[mid].top <= y) lo = mid + 1;
      else hi = mid;
    }
    const point = points[Math.max(0, lo - 1)];
    const {entry, link} = point;
    if (entry !== activeEntry || link !== activeLink) {
      if (activeEntry) activeEntry.classList.remove('is-current-section');
      if (activeLink) {
        activeLink.classList.remove('is-current');
        activeLink.removeAttribute('aria-current');
      }
      entry.classList.add('is-current-section');
      link.classList.add('is-current');
      link.setAttribute('aria-current', 'location');
      activeEntry = entry;
      activeLink = link;
      const chapter = entry.querySelector('.toc-chapter');
      chapterLabel.textContent = chapter.textContent;
      detailLabel.textContent = link === chapter ? '' : link.textContent;
    }
    const group = entry.closest('details');
    if (group) group.open = true;

    // Scroll only the sidebar, never the reading page or keyboard focus.
    if (desktop.matches) {
      const bounds = nav.getBoundingClientRect();
      const banner = nav.querySelector('.reading-location').getBoundingClientRect();
      const item = link.getBoundingClientRect();
      const top = Math.max(bounds.top, banner.bottom) + 14;
      const bottom = bounds.bottom - 18;
      if (item.top < top) nav.scrollTop += item.top - top;
      else if (item.bottom > bottom) nav.scrollTop += item.bottom - bottom;
    }
  }

  function schedule() {
    if (!frame) frame = requestAnimationFrame(update);
  }
  document.documentElement.classList.add('toc-enhanced');
  window.addEventListener('scroll', schedule, {passive:true});
  window.addEventListener('resize', measure);
  window.addEventListener('hashchange', schedule);
  window.addEventListener('pageshow', measure);
  main.addEventListener('toggle', measure, true);
  if (document.fonts) document.fonts.ready.then(measure);
  measure();

  // The native links and source disclosures still work without JavaScript.
  const printOpened = [];
  window.addEventListener('beforeprint', () => {
    for (const item of main.querySelectorAll('details.source-code:not([open]),details.asset-index:not([open]),details.record-details:not([open])')) {
      printOpened.push(item);
      item.open = true;
    }
  });
  window.addEventListener('afterprint', () => {
    for (const item of printOpened.splice(0)) item.open = false;
    measure();
  });
})();
