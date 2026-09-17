/* Interactive line picker for the Profile page.
 *
 * A <div class="wb-picker" data-...> wraps an <img> (a matplotlib scatter) and
 * an <svg> overlay. Clicking inside the plot axes adds a waypoint; the polyline
 * + markers are drawn on the SVG and the UTM coordinates are written as JSON
 * into a hidden <input> the form submits. Coordinate math only — the actual
 * point selection + profile plot happen server-side (integrate).
 */
(function () {
  "use strict";

  function num(el, name) { return parseFloat(el.dataset[name]); }

  function initPicker(root) {
    if (!root || root._wbInit) return;
    root._wbInit = true;

    const img = root.querySelector("img");
    const svg = root.querySelector("svg");
    const hidden = document.getElementById(root.dataset.target || "profile-points");
    const countEl = root.querySelector(".wb-picker-count");
    const meta = {
      axL: num(root, "axL"), axR: num(root, "axR"),
      axB: num(root, "axB"), axT: num(root, "axT"),
      x0: num(root, "x0"), x1: num(root, "x1"),
      y0: num(root, "y0"), y1: num(root, "y1"),
    };
    let points = [];

    // img fraction (0..1 from top-left) <-> data (UTM)
    function fracToData(fx, fy) {
      const u = (fx - meta.axL) / (meta.axR - meta.axL);
      const v = ((1 - fy) - meta.axB) / (meta.axT - meta.axB);
      return [meta.x0 + u * (meta.x1 - meta.x0), meta.y0 + v * (meta.y1 - meta.y0)];
    }
    function dataToUnits(dx, dy) {           // -> svg viewBox units (0..1000)
      const u = (dx - meta.x0) / (meta.x1 - meta.x0);
      const v = (dy - meta.y0) / (meta.y1 - meta.y0);
      const fx = meta.axL + u * (meta.axR - meta.axL);
      const fyB = meta.axB + v * (meta.axT - meta.axB);
      return [fx * 1000, (1 - fyB) * 1000];
    }
    function inside(fx, fy) {
      return fx >= meta.axL && fx <= meta.axR &&
             (1 - fy) >= meta.axB && (1 - fy) <= meta.axT;
    }

    function redraw() {
      const pts = points.map(function (p) { return dataToUnits(p[0], p[1]); });
      let s = "";
      if (pts.length > 1) {
        s += '<polyline points="' + pts.map(function (q) { return q[0] + "," + q[1]; }).join(" ") +
             '" fill="none" stroke="#ec3013" stroke-width="3" vector-effect="non-scaling-stroke"/>';
      }
      pts.forEach(function (q, i) {
        s += '<circle cx="' + q[0] + '" cy="' + q[1] + '" r="6" fill="#ec3013" stroke="#fff" stroke-width="2"/>';
        s += '<text x="' + (q[0] + 9) + '" y="' + (q[1] - 8) + '" font-size="20" fill="#201e1d">' + (i + 1) + "</text>";
      });
      svg.innerHTML = s;
      if (hidden) hidden.value = JSON.stringify(points);
      if (countEl) countEl.textContent = points.length + " point" + (points.length === 1 ? "" : "s");
    }

    root.addEventListener("click", function (e) {
      const r = img.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width;
      const fy = (e.clientY - r.top) / r.height;
      if (fx < 0 || fx > 1 || fy < 0 || fy > 1 || !inside(fx, fy)) return;
      points.push(fracToData(fx, fy));
      redraw();
    });

    root.querySelectorAll("[data-picker-clear]").forEach(function (b) {
      b.addEventListener("click", function () { points = []; redraw(); });
    });
    root.querySelectorAll("[data-picker-undo]").forEach(function (b) {
      b.addEventListener("click", function () { points.pop(); redraw(); });
    });

    redraw();
  }

  function initAll() {
    document.querySelectorAll(".wb-picker").forEach(initPicker);
  }
  document.addEventListener("DOMContentLoaded", initAll);
  // htmx swaps: init any freshly-inserted picker
  document.body.addEventListener("htmx:afterSwap", initAll);
  window.wbInitPickers = initAll;
})();
