// Shared navigation, quiz feedback, and scroll reveal for fingerprint pages.
document.querySelectorAll('.quiz-option').forEach(b => b.addEventListener('click', () => { document.querySelector('#quizOutput').textContent = b.dataset.answer === 'correct' ? 'Correct. The system compares extracted features with enrolled templates.' : 'Not quite. It compares fingerprint features, then looks up the linked ID.' }));
// Gently reveal educational sections; accessible fallback if observer unsupported.
if ('IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const items = document.querySelectorAll('.feature,.use-card,.science-card,.detail-row,.faq-list details,.section-heading');
  const ob = new IntersectionObserver(entries => entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in-view'); ob.unobserve(e.target) } }), { threshold: .08 });
  items.forEach(el => { el.classList.add('reveal-on-scroll'); ob.observe(el) });
}

'use strict';
(() => {
  const toggle = document.getElementById('hawiyaNavToggle');
  const links = document.getElementById('hawiyaNavLinks');
  if (!toggle || !links) return;
  toggle.addEventListener('click', () => {
    const opened = links.classList.toggle('nav-open');
    toggle.setAttribute('aria-expanded', String(opened));
  });
  links.querySelectorAll('a').forEach(a => a.addEventListener('click', () => {
    links.classList.remove('nav-open');
    toggle.setAttribute('aria-expanded', 'false');
  }));
})();
