// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://FredDsR.github.io/arandu',
  base: '/arandu',
  integrations: [
    starlight({
      title: 'Arandu',
      description:
        'Composable pipelines for audio/video transcription, QA generation, and knowledge graph construction.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/FredDsR/arandu',
        },
      ],
      sidebar: [
        { label: 'Getting Started', link: '/getting-started/' },
        { label: 'Configuration', link: '/configuration/' },
        {
          label: 'Guides',
          items: [
            { label: 'Transcription', link: '/guides/transcription/' },
            {
              label: 'Transcription Validation',
              link: '/guides/transcription-validation/',
            },
            { label: 'KG Construction', link: '/guides/kg-construction/' },
            {
              label: 'CEP QA Generation',
              link: '/guides/cep-qa-generation/',
            },
            { label: 'QA Generation', link: '/guides/qa-generation/' },
            { label: 'Evaluation', link: '/guides/evaluation/' },
          ],
        },
        {
          label: 'Reference',
          items: [{ label: 'CLI Reference', link: '/reference/cli/' }],
        },
        {
          label: 'Development',
          items: [
            { label: 'Architecture', link: '/development/architecture/' },
            { label: 'Schemas', link: '/development/schemas/' },
            { label: 'Testing', link: '/development/testing/' },
            { label: 'CI/CD', link: '/development/ci-cd/' },
            { label: 'Dependencies', link: '/development/dependencies/' },
          ],
        },
        {
          label: 'Deployment',
          items: [{ label: 'Docker', link: '/deployment/docker/' }],
        },
      ],
      // Client-side Mermaid rendering (zero server-side dependencies)
      head: [
        {
          tag: 'script',
          attrs: { type: 'module' },
          content: `
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
            mermaid.initialize({ startOnLoad: false, theme: 'neutral' });
            function renderMermaid() {
              const nodes = document.querySelectorAll('pre code.language-mermaid');
              nodes.forEach((code, i) => {
                const div = document.createElement('div');
                div.className = 'mermaid not-content';
                div.id = 'mermaid-' + Date.now() + '-' + i;
                div.textContent = code.textContent;
                code.parentElement.replaceWith(div);
              });
              if (nodes.length) mermaid.run({ querySelector: '.mermaid' });
            }
            document.addEventListener('astro:page-load', renderMermaid);
            renderMermaid();
          `,
        },
      ],
      editLink: {
        baseUrl:
          'https://github.com/FredDsR/arandu/edit/main/docs-site/',
      },
    }),
  ],
});
