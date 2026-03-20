import katex from 'katex';
import 'katex/dist/katex.min.css';

function renderMath(text) {
  if (!text) return '';
  let result = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) =>
    katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false })
  );
  result = result.replace(/\$([^\$\n]+?)\$/g, (_, tex) =>
    katex.renderToString(tex.trim(), { displayMode: false, throwOnError: false })
  );
  result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  return result;
}

export default function MathText({ children, as: Tag = 'div', style, className }) {
  const html = renderMath(typeof children === 'string' ? children : '');
  return <Tag className={className} style={style} dangerouslySetInnerHTML={{ __html: html }} />;
}
