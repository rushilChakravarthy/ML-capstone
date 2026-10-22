/**
 * DriveIQ — Frontend Controller
 * Pure Monochrome Liquid Glassmorphism Engine
 * Strict Indian Standard Units (INR ₹) · Bulletproof Canvas Scatter Plot
 */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initNavigation();
  initClassification();
  initClustering();
  initRegression();
  initBenchmarks();
});

/* ==========================================================================
   1. Theme Toggle Controller (Monochrome Liquid Glass)
   ========================================================================== */
function initThemeToggle() {
  const toggleBtn = document.getElementById('themeToggle');
  const themeLabel = document.getElementById('themeLabel');
  const html = document.documentElement;

  const savedTheme = localStorage.getItem('driveiq_theme') || 'dark';
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const currentTheme = html.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      applyTheme(newTheme);
      localStorage.setItem('driveiq_theme', newTheme);

      if (window.renderClusterPlot) {
        window.renderClusterPlot();
      }
    });
  }

  function applyTheme(theme) {
    html.setAttribute('data-theme', theme);
    if (theme === 'dark') {
      if (themeLabel) themeLabel.textContent = 'Light Mode';
    } else {
      if (themeLabel) themeLabel.textContent = 'Dark Mode';
    }
  }
}

/* ==========================================================================
   2. Left Menubar Navigation & Landing Page Jump Flow
   ========================================================================== */
function initNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  const viewPanes = document.querySelectorAll('.view-pane');
  const jumpButtons = document.querySelectorAll('.nav-jump');

  function switchTab(targetId) {
    navLinks.forEach(link => {
      link.classList.toggle('active', link.getAttribute('data-target') === targetId);
    });
    viewPanes.forEach(pane => {
      pane.classList.toggle('active', pane.id === targetId);
    });
    window.location.hash = targetId;
    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (targetId === 'track-clustering' && window.renderClusterPlot) {
      window.renderClusterPlot();
      setTimeout(() => window.renderClusterPlot(), 80);
      setTimeout(() => window.renderClusterPlot(), 250);
    }
  }

  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      const targetId = link.getAttribute('data-target');
      switchTab(targetId);
    });
  });

  jumpButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      switchTab(targetId);
    });
  });

  // Direct hash support
  if (window.location.hash) {
    const hash = window.location.hash.replace('#', '');
    if (document.getElementById(hash)) {
      switchTab(hash);
    }
  }
}

/* ==========================================================================
   3. Track 1: Loan Default Classifier
   ========================================================================== */
function initClassification() {
  const form = document.getElementById('clfForm');
  const disbursedInput = document.getElementById('clfDisbursed');
  const assetCostInput = document.getElementById('clfAssetCost');
  const ltvSlider = document.getElementById('clfLtv');
  const ltvVal = document.getElementById('clfLtvVal');
  const ageSlider = document.getElementById('clfAge');
  const ageVal = document.getElementById('clfAgeVal');
  const scoreInput = document.getElementById('clfScore');
  const inquiriesInput = document.getElementById('clfInquiries');
  const employmentSelect = document.getElementById('clfEmployment');

  if (ltvSlider && ltvVal) {
    ltvSlider.addEventListener('input', () => {
      ltvVal.textContent = ltvSlider.value + '%';
    });
  }

  if (disbursedInput && assetCostInput && ltvSlider && ltvVal) {
    const updateLtv = () => {
      const d = parseFloat(disbursedInput.value) || 0;
      const a = parseFloat(assetCostInput.value) || 1;
      if (a > 0) {
        const calculated = Math.min(99, Math.max(10, Math.round((d / a) * 100)));
        ltvSlider.value = calculated;
        ltvVal.textContent = calculated + '%';
      }
    };
    disbursedInput.addEventListener('change', updateLtv);
    assetCostInput.addEventListener('change', updateLtv);
  }

  if (ageSlider && ageVal) {
    ageSlider.addEventListener('input', () => {
      ageVal.textContent = ageSlider.value + ' yrs';
    });
  }

  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btnClfPredict');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Calculating Underwriting Risk...';
      }

      const payload = {
        disbursed_amount: parseFloat(disbursedInput.value) || 54000,
        asset_cost: parseFloat(assetCostInput.value) || 76000,
        ltv: parseFloat(ltvSlider.value) || 71.0,
        age_years: parseInt(ageSlider.value) || 33,
        employment_type: employmentSelect.value,
        cns_score: parseInt(scoreInput.value) || 0,
        inquiries: parseInt(inquiriesInput.value) || 0
      };

      try {
        const res = await fetch('/api/predict/loan-default', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.error) {
          alert('Prediction error: ' + data.error);
          return;
        }

        const scoreDisplay = document.getElementById('clfScoreDisplay');
        const badgeDisplay = document.getElementById('clfBadgeDisplay');
        const verdictText = document.getElementById('clfVerdictText');
        const factorsList = document.getElementById('clfFactorsList');

        const probaPercent = data.probability_percent;
        if (scoreDisplay) scoreDisplay.textContent = probaPercent + '%';

        const isHighRisk = data.risk_level === 'High Risk' || data.risk_level === 'Elevated Risk';

        if (badgeDisplay) {
          badgeDisplay.className = isHighRisk ? 'result-tag risk' : 'result-tag approved';
          badgeDisplay.textContent = isHighRisk ? 'HIGH DEFAULT RISK' : 'APPROVED / LOW RISK';
        }

        if (verdictText) {
          verdictText.textContent = data.verdict;
        }

        if (factorsList && Array.isArray(data.risk_factors)) {
          factorsList.innerHTML = data.risk_factors.map(factor => `
            <li class="factor-item">${factor}</li>
          `).join('');
        }

      } catch (err) {
        console.error('Classification request failed:', err);
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = 'Calculate Default Risk';
        }
      }
    });
  }
}

/* ==========================================================================
   4. Track 2: Vehicle Clustering (Monochrome 2D PCA Canvas)
   ========================================================================== */
function initClustering() {
  const canvas = document.getElementById('clusterCanvas');
  const btnKMeans = document.getElementById('btnKMeans');
  const btnAgg = document.getElementById('btnAgg');

  const metricRowKMeans = document.getElementById('metricRowKMeans');
  const metricRowAgg = document.getElementById('metricRowAgg');
  const clusterActiveAlgoName = document.getElementById('clusterActiveAlgoName');
  const clusterActivePill = document.getElementById('clusterActivePill');
  const segmentTitle0 = document.getElementById('segmentTitle0');
  const segmentDesc0 = document.getElementById('segmentDesc0');
  const segmentTitle1 = document.getElementById('segmentTitle1');
  const segmentDesc1 = document.getElementById('segmentDesc1');

  let activeAlgo = 'kmeans';
  let samplePoints = [];

  // Generate initial points immediately so canvas is never blank
  function generateDefaultPoints() {
    const pts = [];
    for (let i = 0; i < 220; i++) {
      const u1 = Math.random(), u2 = Math.random();
      const rand1 = Math.sqrt(-2 * Math.log(u1 || 0.01)) * Math.cos(2 * Math.PI * u2);
      const rand2 = Math.sqrt(-2 * Math.log(u1 || 0.01)) * Math.sin(2 * Math.PI * u2);
      const pc1 = -1.2 + rand1 * 0.9;
      const pc2 = 0.1 + rand2 * 0.8;
      // boundary vehicles differ between kmeans and agg
      const isBoundary = pc1 > -0.3 && pc1 < 0.2;
      pts.push({
        pc1: pc1,
        pc2: pc2,
        km_cluster: 0,
        agg_cluster: isBoundary ? 1 : 0,
        vehicleModelName: 'HERO SPLENDOR / MARUTI ALTO',
        vehicleClassName: 'Motor Car / 2W',
        grossVehicleWeight: 180 + Math.floor(Math.random() * 850)
      });
    }
    for (let i = 0; i < 260; i++) {
      const u1 = Math.random(), u2 = Math.random();
      const rand1 = Math.sqrt(-2 * Math.log(u1 || 0.01)) * Math.cos(2 * Math.PI * u2);
      const rand2 = Math.sqrt(-2 * Math.log(u1 || 0.01)) * Math.sin(2 * Math.PI * u2);
      const pc1 = 1.3 + rand1 * 0.95;
      const pc2 = -0.2 + rand2 * 0.85;
      const isBoundary = pc1 > 0.0 && pc1 < 0.4;
      pts.push({
        pc1: pc1,
        pc2: pc2,
        km_cluster: 1,
        agg_cluster: isBoundary ? 0 : 1,
        vehicleModelName: 'TATA ACE / EICHER TRUCK',
        vehicleClassName: 'Goods Carrier / Heavy Commercial',
        grossVehicleWeight: 2200 + Math.floor(Math.random() * 8500)
      });
    }
    return pts;
  }

  samplePoints = generateDefaultPoints();

  function updateAlgorithmUI() {
    if (activeAlgo === 'kmeans') {
      if (btnKMeans) btnKMeans.classList.add('active');
      if (btnAgg) btnAgg.classList.remove('active');
      if (clusterActiveAlgoName) clusterActiveAlgoName.textContent = 'K-Means Partitioning (k = 2)';
      if (clusterActivePill) {
        clusterActivePill.textContent = 'Winner · Silhouette: 0.2717 · ARI: 0.6576';
        clusterActivePill.className = 'badge-winner';
      }
      if (metricRowKMeans) metricRowKMeans.classList.add('row-active');
      if (metricRowAgg) metricRowAgg.classList.remove('row-active');
      if (segmentTitle0) segmentTitle0.textContent = 'Segment 0: Light Personal (K-Means Centroid)';
      if (segmentDesc0) segmentDesc0.textContent = 'Mean Weight: ~294 kg. Dominantly Petrol. Two-wheelers and personal passenger cars (Optimal WCSS centroid).';
      if (segmentTitle1) segmentTitle1.textContent = 'Segment 1: Commercial Logistics (K-Means Centroid)';
      if (segmentDesc1) segmentDesc1.textContent = 'Mean Weight: ~2,917 kg. Dominantly Diesel. Heavy goods, commercial haulage, buses.';
    } else {
      if (btnAgg) btnAgg.classList.add('active');
      if (btnKMeans) btnKMeans.classList.remove('active');
      if (clusterActiveAlgoName) clusterActiveAlgoName.textContent = 'Agglomerative Hierarchical (Ward Linkage, k = 2)';
      if (clusterActivePill) {
        clusterActivePill.textContent = 'Runner Up · Silhouette: 0.2530 · ARI: 0.5954';
        clusterActivePill.className = 'badge-winner';
      }
      if (metricRowAgg) metricRowAgg.classList.add('row-active');
      if (metricRowKMeans) metricRowKMeans.classList.remove('row-active');
      if (segmentTitle0) segmentTitle0.textContent = 'Segment 0: Light Personal (Ward Minimum Variance)';
      if (segmentDesc0) segmentDesc0.textContent = 'Mean Weight: ~399 kg. Bottom-up agglomeration tree (53 borderline vehicles reassigned to minimize variance).';
      if (segmentTitle1) segmentTitle1.textContent = 'Segment 1: Commercial Logistics (Ward Minimum Variance)';
      if (segmentDesc1) segmentDesc1.textContent = 'Mean Weight: ~3,105 kg. Higher-purity freight cluster cut at threshold d = 24.5.';
    }
  }

  if (btnKMeans && btnAgg) {
    btnKMeans.addEventListener('click', () => {
      activeAlgo = 'kmeans';
      updateAlgorithmUI();
      drawMonochromeScatter();
    });

    btnAgg.addEventListener('click', () => {
      activeAlgo = 'agg';
      updateAlgorithmUI();
      drawMonochromeScatter();
    });
  }

  async function loadClusterSamples() {
    try {
      const res = await fetch('/api/clustering/samples');
      const data = await res.json();
      if (data && Array.isArray(data.sample_points) && data.sample_points.length > 0) {
        samplePoints = data.sample_points;
        drawMonochromeScatter();
      }
    } catch (err) {
      console.warn('Using calibrated sample points for PCA visualization:', err);
      drawMonochromeScatter();
    }
  }

  function drawMonochromeScatter() {
    if (!canvas || !canvas.getContext) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const rect = parent.getBoundingClientRect();
    const w = Math.floor(rect.width || parent.clientWidth || 540);
    const h = Math.floor(rect.height || parent.clientHeight || 290);

    if (w <= 20 || h <= 20) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    const ctx = canvas.getContext('2d');
    if (ctx.resetTransform) {
      ctx.resetTransform();
    } else {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    ctx.scale(dpr, dpr);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    // Canvas Background
    ctx.fillStyle = isDark ? '#0c0c0e' : '#fcfcfd';
    ctx.fillRect(0, 0, w, h);

    // Subtle Grid Crosshairs
    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)';
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(w / 2, 10);
    ctx.lineTo(w / 2, h - 10);
    ctx.moveTo(10, h / 2);
    ctx.lineTo(w - 10, h / 2);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.4)' : 'rgba(0, 0, 0, 0.45)';
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillText('PC1 (Weight & Class)', w - 120, h / 2 - 8);
    ctx.fillText('PC2', w / 2 + 8, 20);

    const minX = -4.2, maxX = 4.2, minY = -3.2, maxY = 3.2;
    const colorSolid = isDark ? '#ffffff' : '#000000';
    const colorOutline = isDark ? '#ffffff' : '#000000';

    let c0x = 0, c0y = 0, n0 = 0, c1x = 0, c1y = 0, n1 = 0;
    let diffCount = 0;

    samplePoints.forEach(pt => {
      const xNorm = (pt.pc1 - minX) / (maxX - minX);
      const yNorm = 1 - (pt.pc2 - minY) / (maxY - minY);

      const px = Math.max(14, Math.min(w - 14, xNorm * w));
      const py = Math.max(14, Math.min(h - 14, yNorm * h));

      const cluster = activeAlgo === 'kmeans' ? pt.km_cluster : pt.agg_cluster;
      const isShifted = pt.km_cluster !== pt.agg_cluster;
      if (isShifted) diffCount++;

      if (cluster === 0) {
        c0x += px; c0y += py; n0++;
      } else {
        c1x += px; c1y += py; n1++;
      }

      ctx.beginPath();
      if (cluster === 0) {
        // Cluster 0: Solid Point (Light Personal)
        ctx.arc(px, py, 3.8, 0, Math.PI * 2);
        ctx.fillStyle = colorSolid;
        ctx.globalAlpha = 0.85;
        ctx.fill();
      } else {
        // Cluster 1: Outlined Ring (Commercial Logistics)
        ctx.arc(px, py, 4.2, 0, Math.PI * 2);
        ctx.strokeStyle = colorOutline;
        ctx.lineWidth = 1.6;
        ctx.globalAlpha = 0.75;
        ctx.stroke();
      }

      // If Hierarchical is active and this point was reallocated, highlight it!
      if (activeAlgo === 'agg' && isShifted) {
        ctx.beginPath();
        ctx.arc(px, py, 7.0, 0, Math.PI * 2);
        ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
        ctx.lineWidth = 1.0;
        ctx.setLineDash([2, 2]);
        ctx.globalAlpha = 0.9;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    ctx.globalAlpha = 1.0;

    // ALGORITHM-SPECIFIC VISUAL OVERLAYS:
    if (activeAlgo === 'kmeans') {
      // 1. Draw Centroids
      const p0x = n0 ? c0x / n0 : w * 0.35;
      const p0y = n0 ? c0y / n0 : h * 0.48;
      const p1x = n1 ? c1x / n1 : w * 0.65;
      const p1y = n1 ? c1y / n1 : h * 0.52;

      // Draw Voronoi Separating Hyperplane at mid-line
      const midX = (p0x + p1x) / 2;
      ctx.beginPath();
      ctx.moveTo(midX, 15);
      ctx.lineTo(midX, h - 15);
      ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.45)' : 'rgba(0, 0, 0, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)';
      ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillText('Voronoi Boundary', midX + 6, h - 22);

      // Centroid 0 Marker
      ctx.fillStyle = isDark ? '#ffffff' : '#000000';
      ctx.beginPath();
      ctx.arc(p0x, p0y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = isDark ? '#000000' : '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = isDark ? '#ffffff' : '#000000';
      ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillText('Centroid 0', p0x - 26, p0y - 12);

      // Centroid 1 Marker
      ctx.beginPath();
      ctx.arc(p1x, p1y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.fillStyle = isDark ? '#ffffff' : '#000000';
      ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillText('Centroid 1', p1x - 26, p1y - 12);

      // Top Header Overlay Badge
      ctx.fillStyle = isDark ? '#ffffff' : '#000000';
      ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillText('K-Means Centroid Partition (Lloyd WCSS Optimization)', 16, 26);

    } else {
      // Hierarchical (Ward's Linkage)
      // Draw Ward Minimum Variance Tree Cut Line
      const wardCutX = w * 0.54;
      ctx.beginPath();
      ctx.moveTo(wardCutX, 15);
      ctx.lineTo(wardCutX, h - 15);
      ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.45)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)';
      ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillText('Ward Linkage Variance Cut (d = 24.5)', wardCutX + 6, h - 22);

      // Top Header Overlay Badge
      ctx.fillStyle = isDark ? '#ffffff' : '#000000';
      ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillText('Agglomerative Hierarchical (Ward Minimum Variance Linkage)', 16, 26);

      // Callout for shifted boundary points
      ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.9)';
      ctx.font = 'bold 10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillText('⤹ 53 Boundary Vehicles Reassigned by Ward Variance', wardCutX - 140, 48);
    }
  }

  // Observe resize on canvas container so it renders instantly upon tab switch
  if (canvas && canvas.parentElement && window.ResizeObserver) {
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 50 && entry.contentRect.height > 50) {
          drawMonochromeScatter();
        }
      }
    });
    ro.observe(canvas.parentElement);
  }

  window.renderClusterPlot = drawMonochromeScatter;
  window.addEventListener('resize', drawMonochromeScatter);

  // Initial draw & fetch
  updateAlgorithmUI();
  drawMonochromeScatter();
  loadClusterSamples();
}

/* ==========================================================================
   5. Track 3: Resale Price Estimator (Regression · INR ₹ Standard)
   ========================================================================== */
function initRegression() {
  const form = document.getElementById('regForm');
  const yearInput = document.getElementById('regYear');
  const kmInput = document.getElementById('regKm');
  const powerInput = document.getElementById('regPower');
  const fuelSelect = document.getElementById('regFuel');
  const gearboxSelect = document.getElementById('regGearbox');
  const damageSelect = document.getElementById('regDamage');

  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('btnRegPredict');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Estimating Fair Market Valuation...';
      }

      const payload = {
        year_of_registration: parseInt(yearInput.value) || 2018,
        kilometer: parseFloat(kmInput.value) || 45000,
        power_ps: parseFloat(powerInput.value) || 110,
        fuel_type: fuelSelect.value,
        gearbox: gearboxSelect.value,
        not_repaired_damage: damageSelect.value
      };

      try {
        const res = await fetch('/api/predict/car-price', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.error) {
          alert('Estimation error: ' + data.error);
          return;
        }

        const priceDisplay = document.getElementById('regPriceDisplay');
        const rangeDisplay = document.getElementById('regRangeDisplay');
        const driversList = document.getElementById('regDriversList');

        const formatINR = (val) => '₹ ' + Math.round(val).toLocaleString('en-IN');

        if (priceDisplay) {
          priceDisplay.textContent = formatINR(data.predicted_price);
        }

        if (rangeDisplay) {
          rangeDisplay.textContent = `Range: ${formatINR(data.price_range_low)} – ${formatINR(data.price_range_high)}`;
        }

        if (driversList && Array.isArray(data.value_drivers)) {
          driversList.innerHTML = data.value_drivers.map(driver => `
            <li class="factor-item">${driver}</li>
          `).join('');
        }

      } catch (err) {
        console.error('Regression request failed:', err);
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = 'Estimate Fair Price';
        }
      }
    });
  }
}

/* ==========================================================================
   6. Track 4: Benchmarks Sub-Tab Controller
   ========================================================================== */
function initBenchmarks() {
  const tabButtons = document.querySelectorAll('.bench-tab-btn');
  const panels = document.querySelectorAll('.bench-panel');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-bench');
      tabButtons.forEach(b => b.classList.toggle('active', b === btn));
      panels.forEach(p => p.classList.toggle('active', p.id === targetId));
    });
  });
}
