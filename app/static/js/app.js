// Generic client helpers - loads model list for pages that need it.
window.fetchJSON = function (url, options) {
  return fetch(url, options).then(function (r) { return r.json(); });
};
