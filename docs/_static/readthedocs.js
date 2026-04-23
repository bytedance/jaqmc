function triggerRtdSearch() {
  const event = new CustomEvent("readthedocs-search-show");
  document.dispatchEvent(event);
}

document.addEventListener("DOMContentLoaded", function (event) {
  document.querySelectorAll(".search-button__button").forEach((element) => {
    element.onclick = triggerRtdSearch;
  });
  document.addEventListener("keydown", (e) => {
    const useCommandKey =
      navigator.platform.indexOf("Mac") === 0 ||
      navigator.platform === "iPhone";
    const metaKeyUsed = useCommandKey
      ? e.metaKey && !e.ctrlKey
      : !e.metaKey && e.ctrlKey;
    if (metaKeyUsed && e.key === "k" && !e.shiftKey && !e.altKey) {
      const searchDialog = document.getElementById("pst-search-dialog");
      if (document.contains(searchDialog)) {
        searchDialog.close();
      }
      triggerRtdSearch();
    }
  });
});
