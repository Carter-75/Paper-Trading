const UI = {
  statusDot: document.getElementById('status-dot'),
  statusPing: document.getElementById('status-ping'),
  timestamp: document.getElementById('timestamp'),
  
  btnPause: document.getElementById('btn-pause'),
  btnResume: document.getElementById('btn-resume'),
  btnStop: document.getElementById('btn-stop'),
  
  eqTotal: document.getElementById('eq-total'),
  eqHard1: document.getElementById('eq-hard1'),
  eqHard2: document.getElementById('eq-hard2'),
  eqSoft: document.getElementById('eq-soft'),
  eqHwm: document.getElementById('eq-hwm'),
  
  countdown: document.getElementById('countdown'),
  nextCycle: document.getElementById('next-cycle'),
  lastSymbol: document.getElementById('last-symbol'),
  lastAction: document.getElementById('last-action'),
  confidence: document.getElementById('confidence'),
  
  portfolioBody: document.getElementById('portfolio-body'),
  historyBody: document.getElementById('history-body'),
  logBox: document.getElementById('log-box')
};

let nextCycleTarget = null;

// Utility: Format Currency
const formatMoney = (val) => {
  if (val === undefined || val === null || isNaN(val)) return '---';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(Number(val));
};

// Utility: Format Percent
const formatPercent = (val, dec = 2) => {
  if (val === undefined || val === null || isNaN(val)) return '---';
  return (Number(val) * 100).toFixed(dec) + '%';
};

// Handle Control Buttons
const controlBot = async (action) => {
  if (!confirm(`Are you sure you want to ${action.toUpperCase()} the bot?`)) return;
  
  const buttons = [UI.btnPause, UI.btnResume, UI.btnStop];
  buttons.forEach(b => b.disabled = true);
  
  try {
    const res = await fetch(`/api/control/${action}`, { method: 'POST' });
    const data = await res.json();
    alert(`Response: ${data.status || data.error}`);
  } catch (err) {
    alert("Error communicating with bot. Check if server is running.");
  } finally {
    buttons.forEach(b => b.disabled = false);
    fetchState();
  }
};

// Attach listeners
UI.btnPause.addEventListener('click', () => controlBot('pause'));
UI.btnResume.addEventListener('click', () => controlBot('resume'));
UI.btnStop.addEventListener('click', () => controlBot('stop'));

// Update Countdown
const updateCountdown = () => {
  if (!nextCycleTarget) {
    UI.countdown.textContent = "--s";
    return;
  }
  const now = new Date();
  const diff = Math.max(0, Math.floor((nextCycleTarget - now) / 1000));
  UI.countdown.textContent = diff + "s";
};

// Fetch & Render History
const fetchHistory = async () => {
  try {
    const res = await fetch('/api/history?_=' + Date.now());
    const data = await res.json();
    
    if (Array.isArray(data) && data.length > 0) {
      const sorted = data.slice().reverse();
      UI.historyBody.innerHTML = sorted.map(trade => {
        const isPositive = trade.pnl >= 0;
        const pnlColorClass = isPositive ? 'text-success' : 'text-danger';
        const dateStr = new Date(trade.timestamp).toLocaleString();
        
        return `
          <tr>
            <td style="color: var(--text-muted)">${dateStr}</td>
            <td style="font-weight: 500">${trade.symbol}</td>
            <td class="text-right ${pnlColorClass}">${formatMoney(trade.pnl)}</td>
            <td class="text-right ${pnlColorClass}">${formatPercent(trade.pnl_pct, 2)}</td>
            <td style="color: var(--text-muted); font-size: 0.8125rem;">${trade.reason}</td>
          </tr>
        `;
      }).join('');
    } else {
      UI.historyBody.innerHTML = '<tr><td colspan="5" class="text-center" style="color: var(--text-faint)">No trade history yet</td></tr>';
    }
  } catch (err) {
    console.error("History fetch error", err);
  }
};

// Fetch & Render State
const fetchState = async () => {
  try {
    const res = await fetch('/api/state?_=' + Date.now());
    const data = await res.json();
    
    // Status Indicator
    let statusColor = 'var(--text-faint)';
    let statusTitle = 'Offline / Unknown';
    
    if (data.active_cycle) {
      if (data.restricted_mode) {
        statusColor = 'var(--warning)';
        statusTitle = 'Restricted Mode (Soft Kill Recovery)';
      } else {
        statusColor = 'var(--success)';
        statusTitle = 'Active';
      }
    } else if (data.last_symbol === "PAUSED") {
      statusColor = 'var(--info)';
      statusTitle = 'Paused';
    } else if (data.last_symbol && data.last_symbol.includes("KILL_SWITCH")) {
      statusColor = 'var(--danger)';
      statusTitle = 'KILLED';
    }
    
    UI.statusDot.style.backgroundColor = statusColor;
    UI.statusPing.style.backgroundColor = statusColor;
    UI.statusDot.parentElement.title = statusTitle;
    
    if (statusTitle !== 'Active' && statusTitle !== 'Restricted Mode (Soft Kill Recovery)') {
      UI.statusPing.style.display = 'none';
    } else {
      UI.statusPing.style.display = 'block';
    }

    if (data.timestamp) {
      UI.timestamp.textContent = "Last sync: " + new Date(data.timestamp).toLocaleString();
    }
    
    // Equity Grid
    if (data.equity !== undefined) {
      UI.eqTotal.textContent = formatMoney(data.equity);
      UI.eqHard1.textContent = formatMoney(data.kill_switch_floor_usd);
      
      if (data.hard_kill_2_level !== undefined) UI.eqHard2.textContent = formatMoney(data.hard_kill_2_level);
      if (data.dynamic_floor_level !== undefined) UI.eqSoft.textContent = formatMoney(data.dynamic_floor_level);
      if (data.high_water_mark !== undefined) UI.eqHwm.textContent = formatMoney(data.high_water_mark);
      
      // Cycle & Action
      if (data.next_cycle_time) {
        nextCycleTarget = new Date(data.next_cycle_time);
        UI.nextCycle.textContent = "Next at " + nextCycleTarget.toLocaleTimeString();
      }
      
      UI.lastSymbol.textContent = data.last_symbol || "Scanning...";
      
      const action = (data.last_action || "").toUpperCase();
      UI.lastAction.textContent = action || "---";
      UI.lastAction.className = 'stat-value ' + (action === 'BUY' ? 'text-success' : action === 'SELL' ? 'text-danger' : '');
      
      UI.confidence.textContent = data.last_confidence !== undefined ? formatPercent(data.last_confidence, 0) : '---';
      
      // Portfolio
      if (data.positions && Object.keys(data.positions).length > 0) {
        UI.portfolioBody.innerHTML = Object.entries(data.positions).map(([sym, pos]) => {
          const sl = pos.stop_loss || 0;
          const tp = pos.take_profit || 0;
          const ptp = pos.ptp_executed ? '<span class="status-dot sm" style="background:var(--success); display:inline-block; margin-right:4px;" title="PTP Executed"></span>' : '';
          const pl = pos.unrealized_pl || 0;
          const isPositive = pl >= 0;
          const plColorClass = isPositive ? 'text-success' : 'text-danger';
          
          return `
            <tr>
              <td style="font-weight: 500">${sym}</td>
              <td class="text-right">${Number(pos.qty).toFixed(2)}</td>
              <td class="text-right">${formatMoney(pos.avg_entry)}</td>
              <td class="text-right text-danger">${formatMoney(sl)}</td>
              <td class="text-right text-success">${ptp}${formatMoney(tp)}</td>
              <td class="text-right ${plColorClass}">${formatMoney(pl)}</td>
            </tr>
          `;
        }).join('');
      } else {
        UI.portfolioBody.innerHTML = '<tr><td colspan="6" class="text-center" style="color: var(--text-faint)">No active positions</td></tr>';
      }
    }
  } catch (err) {
    console.error("State fetch error", err);
  }
};

// Fetch & Render Logs
const fetchLogs = async () => {
  try {
    const res = await fetch('/api/logs?_=' + Date.now());
    const data = await res.json();
    
    if (data.logs && data.logs.length > 0) {
      const isScrolledToBottom = UI.logBox.scrollHeight - UI.logBox.clientHeight <= UI.logBox.scrollTop + 50;
      
      UI.logBox.innerHTML = data.logs.map(line => {
        let cls = 'log-line';
        if (line.includes('[ERROR]') || line.includes('ERROR')) cls += ' log-error';
        else if (line.includes('[WARNING]') || line.includes('WARN')) cls += ' log-warning';
        else if (line.includes('SIGNAL')) cls += ' log-signal';
        
        // Escape HTML
        const safeLine = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `<span class="${cls}">${safeLine}</span>`;
      }).join('');
      
      if (isScrolledToBottom) {
        UI.logBox.scrollTop = UI.logBox.scrollHeight;
      }
    }
  } catch (err) {
    console.error("Logs fetch error", err);
  }
};

// Initialize
const init = () => {
  UI.portfolioBody.innerHTML = '<tr><td colspan="6" style="padding: 1rem;"><div class="skeleton skeleton-row"></div><div class="skeleton skeleton-row"></div></td></tr>';
  UI.historyBody.innerHTML = '<tr><td colspan="5" style="padding: 1rem;"><div class="skeleton skeleton-row"></div><div class="skeleton skeleton-row"></div></td></tr>';
  
  fetchState();
  fetchLogs();
  fetchHistory();
  
  setInterval(fetchState, 2000);
  setInterval(fetchLogs, 2000);
  setInterval(fetchHistory, 5000);
  setInterval(updateCountdown, 1000);
};

document.addEventListener('DOMContentLoaded', init);
