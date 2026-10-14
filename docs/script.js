'use strict';

// All asset paths are relative so the page works under a GitHub Pages repository path.
document.querySelectorAll('.comparison-range').forEach((range) => {
  const update = () => {
    const value = Number(range.value);
    range.closest('.comparison').style.setProperty('--position', `${value}%`);
    range.setAttribute('aria-valuetext', `${value} percent input, ${100 - value} percent GeoMAR`);
  };
  range.addEventListener('input', update);
  update();
});

const datasets = {
  lfw: { label: 'LFW-Test' },
  webphoto: { label: 'WebPhoto-Test' },
  wider: { label: 'WIDER-Test' },
  celeba: { label: 'CelebA-Test' }
};
const tabs = [...document.querySelectorAll('[data-dataset]')];
const grid = document.getElementById('baseline-grid');
const panel = document.getElementById('comparison-panel');

function selectDataset(tab) {
  const key = tab.dataset.dataset;
  const dataset = datasets[key];
  tabs.forEach((item) => {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  panel.setAttribute('aria-labelledby', tab.id);
  const methods = [['input', 'Input'], ['codeformer', 'CodeFormer'], ['daefr', 'DAEFR'], ['geomar', 'GeoMAR']];
  if (key === 'celeba') methods.push(['gt', 'Ground truth']);
  grid.classList.toggle('with-gt', key === 'celeba');
  grid.replaceChildren(...methods.map(([method, label]) => {
    const figure = document.createElement('figure');
    if (method === 'geomar') figure.className = 'ours';
    const img = document.createElement('img');
    img.src = `assets/${key}-${method}.webp`;
    img.alt = `${label} on ${dataset.label}`;
    img.width = 512;
    img.height = 512;
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    if (method === 'geomar') {
      const badge = document.createElement('span');
      badge.textContent = 'Ours';
      caption.append(badge);
    }
    figure.append(img, caption);
    return figure;
  }));
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectDataset(tab));
  tab.addEventListener('keydown', (event) => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    tabs[next].focus();
    selectDataset(tabs[next]);
  });
});

const copyButton = document.getElementById('copy-citation');
const copyStatus = document.getElementById('copy-status');
let resetCopyTimer;
copyButton.addEventListener('click', async () => {
  const text = document.getElementById('bibtex').textContent;
  let copied = false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch (_) { /* Fall back for file URLs or restricted clipboard permissions. */ }
  if (!copied) {
    const input = document.createElement('textarea');
    input.value = text;
    input.style.cssText = 'position:fixed;left:-9999px;top:0';
    input.setAttribute('aria-label', 'Citation to copy');
    document.body.append(input);
    input.select();
    try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
    input.remove();
    copyButton.focus({ preventScroll: true });
  }
  clearTimeout(resetCopyTimer);
  copyButton.textContent = copied ? 'Copied ✓' : 'Select citation';
  copyStatus.textContent = copied ? 'BibTeX copied to clipboard.' : 'Automatic copying is unavailable. The citation is selected; use Ctrl+C or Command+C.';
  if (!copied) {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
  }
  resetCopyTimer = setTimeout(() => {
    copyButton.textContent = 'Copy citation';
    copyStatus.textContent = '';
  }, 3500);
});
