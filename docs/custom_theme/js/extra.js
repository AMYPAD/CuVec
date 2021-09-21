/**
 * replacement for mkdocs.yml::theme.features: [content.tabs.link]
 * https://github.com/facelessuser/pymdown-extensions/issues/1456#issuecomment-923155517
 */
const tabs = e => {
  if (e.target.matches('.tabbed-set > input[type=radio]')) {
    const label = e.target.nextSibling.innerHTML;
    const labels = document.querySelectorAll('.tabbed-set > label');
    labels.forEach(el => {
      if (el.innerHTML === label) el.previousSibling.checked = true;
    });
  }
};
document.addEventListener("click", tabs);
