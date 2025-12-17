import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import {
  Upload,
  Play,
  FileText,
  Activity,
  DollarSign,
  Trash2,
  Users,
  AlertCircle,
  CheckCircle2,
  X,
  ChevronDown,
} from 'lucide-react';

const API_URL = import.meta.env.PROD ? '' : 'http://localhost:8000';
const STRATEGIES = ['Greedy', 'Near-Opt', 'RWES_T', 'Yomna', 'Hybrid'];

function App() {
  const [config, setConfig] = useState({
    num_days: 10,
    num_stores: 15,
    num_customers: 70,
    use_uploaded_data: false,
    stores_filename: null,
    customers_filename: null,
    skip_anan: true,
  });

  const [status, setStatus] = useState('idle'); // idle, uploading, running, completed, error
  const [resultsId, setResultsId] = useState(null);
  const [results, setResults] = useState(null);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [toasts, setToasts] = useState([]);

  // View controls
  const [primaryStrategy, setPrimaryStrategy] = useState('Hybrid');
  const [timeWindow, setTimeWindow] = useState('all'); // all | first_half | second_half
  const [focusedChart, setFocusedChart] = useState(null); // null | 'revenue' | 'waste'

  // Close modal on Esc
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') setFocusedChart(null);
    };
    if (focusedChart) window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [focusedChart]);

  // Toasts
  const showToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => removeToast(id), 5000);
  };

  const removeToast = (id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  // Upload handler
  const handleFileUpload = async (event, type) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setStatus('uploading');
    setLoadingMsg(`Uploading ${type}...`);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setConfig((prev) => ({
        ...prev,
        use_uploaded_data: true,
        [`${type}_filename`]: response.data.filename,
      }));

      setStatus('idle');
      showToast(
        `${type === 'stores' ? 'Stores' : 'Customers'} CSV uploaded successfully!`,
        'success'
      );
    } catch (error) {
      console.error('Upload failed:', error);
      setStatus('error');
      showToast('Upload failed. Please check the console.', 'error');
    }
  };

  // Run simulation
  const runSimulation = async () => {
    setStatus('running');
    setLoadingMsg('Starting simulation...');

    try {
      const response = await axios.post(`${API_URL}/run`, config);
      setResultsId(response.data.results_id);
      pollResults(response.data.results_id);
      showToast('Simulation started...', 'info');
    } catch (error) {
      console.error('Simulation start failed:', error);
      setStatus('error');
      showToast('Failed to start simulation.', 'error');
    }
  };

  // Poll for results
  const pollResults = async (id) => {
    const interval = setInterval(async () => {
      try {
        setLoadingMsg('Simulation in progress... this may take a moment.');
        const response = await axios.get(`${API_URL}/results/${id}`);

        if (response.data.status === 'completed') {
          clearInterval(interval);
          setResults(response.data.data);
          setStatus('completed');
          showToast('Simulation completed successfully!', 'success');
        } else if (response.data.status === 'failed') {
          clearInterval(interval);
          setStatus('error');
          showToast(`Simulation failed: ${response.data.error}`, 'error');
        }
      } catch (error) {
        console.log('Polling...');
      }
    }, 2000);
  };

  // Derived insights
  const insights = results?.comparison
    ? deriveInsights(results.comparison, primaryStrategy)
    : null;

  // Filtered time-series data
  const filteredSeriesData = (metricKey) =>
    filterByWindow(prepareMultiStrategyData(results, metricKey), timeWindow);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-50 relative overflow-hidden">
      {/* Background accents */}
      <div className="pointer-events-none absolute inset-0 opacity-60">
        <div className="absolute -top-32 -right-24 h-72 w-72 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="absolute -bottom-40 -left-32 h-80 w-80 rounded-full bg-sky-500/15 blur-3xl" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(148,163,184,0.22),_transparent_60%)]" />
      </div>

      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            {...toast}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-10">
        {/* Header */}
        <header className="mb-8 lg:mb-10">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-200 mb-3">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Food Waste Simulation Lab
              </div>
              <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-slate-50">
                Food Waste Reduction{' '}
                <span className="text-emerald-400">Simulation</span>
              </h1>
              <p className="mt-2 text-sm md:text-base text-slate-300 max-w-2xl">
                Configure a virtual supermarket network and compare strategies
                that trade off
                <span className="font-medium text-emerald-200"> revenue</span>,
                <span className="font-medium text-rose-200"> waste</span>, and
                <span className="font-medium text-sky-200">
                  {' '}
                  customer satisfaction
                </span>
                .
              </p>
            </div>

            <div className="flex flex-col items-start lg:items-end gap-3">
              <div className="flex items-center gap-2 rounded-xl border border-slate-700/70 bg-slate-900/70 px-3 py-2 text-xs text-slate-300 shadow-lg shadow-black/30">
                <Activity
                  className={`h-4 w-4 ${
                    status === 'running'
                      ? 'text-emerald-400 animate-spin'
                      : status === 'error'
                      ? 'text-rose-400'
                      : 'text-slate-500'
                  }`}
                />
                <span className="uppercase tracking-wide">
                  {status === 'running'
                    ? 'Running simulation'
                    : status === 'uploading'
                    ? 'Uploading data'
                    : status === 'completed'
                    ? 'Last run completed'
                    : status === 'error'
                    ? 'Error – check logs'
                    : 'Ready'}
                </span>
              </div>
              <div className="flex flex-wrap gap-2 text-[11px] text-slate-400">
                <span className="rounded-full bg-slate-900/70 border border-slate-700/70 px-2 py-1">
                  API: {API_URL}
                </span>
                <span className="rounded-full bg-slate-900/70 border border-slate-700/70 px-2 py-1">
                  Algorithms: Greedy · Near‑Opt · RWES_T · Yomna · Hybrid
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Main */}
        <main className="space-y-8">
          {/* Config + Snapshot */}
          <section className="grid gap-6 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1.1fr)]">
            {/* Configuration */}
            <section className="bg-slate-900/80 rounded-2xl border border-slate-800/80 shadow-2xl shadow-black/40 p-6 md:p-7 backdrop-blur transition-all duration-300 hover:border-emerald-400/60 hover:shadow-emerald-500/30">
              <h2 className="flex items-center gap-2 text-lg font-semibold text-slate-50 mb-6">
                <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-300">
                  <FileText size={18} />
                </span>
                Simulation Configuration
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">
                <ConfigField
                  label="Simulation days"
                  helper="Longer runs reveal stability and long‑term waste."
                >
                  <input
                    type="number"
                    min="1"
                    className="w-full px-3 py-2.5 rounded-xl border border-slate-700 bg-slate-950/60 text-slate-50 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/60 focus:border-emerald-400 placeholder:text-slate-500"
                    value={config.num_days}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        num_days: parseInt(e.target.value) || 10,
                      })
                    }
                  />
                </ConfigField>

                <ConfigField
                  label="Target stores"
                  helper="Ignored when a stores CSV is provided."
                >
                  <input
                    type="number"
                    min="1"
                    className="w-full px-3 py-2.5 rounded-xl border border-slate-700 bg-slate-950/60 text-slate-50 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/60 focus:border-emerald-400 placeholder:text-slate-500 disabled:bg-slate-900/60 disabled:text-slate-500"
                    value={config.num_stores}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        num_stores: parseInt(e.target.value) || 15,
                      })
                    }
                    disabled={config.use_uploaded_data}
                  />
                </ConfigField>

                <ConfigField
                  label="Target customers"
                  helper="Larger populations increase demand variance."
                >
                  <input
                    type="number"
                    min="1"
                    className="w-full px-3 py-2.5 rounded-xl border border-slate-700 bg-slate-950/60 text-slate-50 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/60 focus:border-emerald-400 placeholder:text-slate-500 disabled:bg-slate-900/60 disabled:text-slate-500"
                    value={config.num_customers}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        num_customers: parseInt(e.target.value) || 70,
                      })
                    }
                    disabled={config.use_uploaded_data}
                  />
                </ConfigField>
              </div>

              <div className="flex flex-col md:flex-row gap-4 mb-5">
                <UploadCard
                  label="Upload Stores CSV"
                  subtitle="Required columns: store_id, capacity, city, …"
                  onChange={(e) => handleFileUpload(e, 'stores')}
                  filename={config.stores_filename}
                  accent="emerald"
                />
                <UploadCard
                  label="Upload Customers CSV"
                  subtitle="Required columns: customer_id, segment, …"
                  onChange={(e) => handleFileUpload(e, 'customers')}
                  filename={config.customers_filename}
                  accent="sky"
                />
              </div>

              <div className="flex items-start gap-2 mb-5 text-xs text-slate-400">
                <AlertCircle className="h-4 w-4 text-amber-300 mt-0.5" />
                <p>
                  When CSVs are provided, the simulator runs on your real
                  network (stores & customers). Without uploads, it generates a
                  synthetic but consistent scenario from the parameters above.
                </p>
              </div>

              <button
                className={`w-full py-3.5 rounded-xl font-semibold text-sm md:text-base flex items-center justify-center gap-2 transition-all duration-200 shadow-lg shadow-emerald-500/30 ${
                  status === 'running' || status === 'uploading'
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-emerald-500 text-slate-950 hover:bg-emerald-400 active:scale-[0.99]'
                }`}
                onClick={runSimulation}
                disabled={status === 'running' || status === 'uploading'}
              >
                {status === 'running' ? (
                  <Activity className="animate-spin h-4 w-4" />
                ) : (
                  <Play size={18} />
                )}
                {status === 'running'
                  ? loadingMsg || 'Running simulation...'
                  : 'Run Simulation'}
              </button>
            </section>

            {/* Snapshot + Controls */}
            <aside className="space-y-4">
              <ScenarioSnapshot config={config} />
              <ControlPanel
                primaryStrategy={primaryStrategy}
                setPrimaryStrategy={setPrimaryStrategy}
                timeWindow={timeWindow}
                setTimeWindow={setTimeWindow}
              />
            </aside>
          </section>

          {/* Results */}
          {status === 'completed' && results && (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
              {/* KPI cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {results.comparison && (
                  <>
                    <KPICard
                      title="Best Revenue"
                      value={getMaxMetric(results.comparison, 'Revenue')}
                      icon={<DollarSign size={28} className="text-white" />}
                      gradient="from-emerald-400 to-emerald-600"
                      glow="shadow-emerald-400/40"
                    />
                    <KPICard
                      title="Lowest Waste"
                      value={getMinMetric(
                        results.comparison,
                        'Waste Bags'
                      )}
                      icon={<Trash2 size={28} className="text-white" />}
                      gradient="from-rose-400 to-rose-600"
                      glow="shadow-rose-400/40"
                    />
                    <KPICard
                      title="Avg Satisfaction"
                      value={getMaxMetric(
                        results.comparison,
                        'Satisfaction',
                        false
                      )}
                      icon={<Users size={28} className="text-white" />}
                      gradient="from-sky-400 to-sky-600"
                      glow="shadow-sky-400/40"
                    />
                  </>
                )}
              </div>

              {/* Charts + insights */}
              <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,2fr)_minmax(260px,0.95fr)] gap-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Revenue */}
                  <ChartCard
                    title="Revenue Trends"
                    badge="Total revenue per day"
                    badgeColor="emerald"
                    onClick={() => setFocusedChart('revenue')}
                  >
                    <div className="h-[420px] w-full">
                      <ResponsiveContainer>
                        <LineChart
                          data={filteredSeriesData('total_revenue')}
                          margin={{
                            top: 10,
                            right: 30,
                            left: 10,
                            bottom: 5,
                          }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            vertical={false}
                            stroke="#e2e8f0"
                          />
                          <XAxis
                            dataKey="day"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#94a3b8' }}
                          />
                          <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#94a3b8' }}
                            tickFormatter={(value) => `$${value}`}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'rgba(255,255,255,0.96)',
                              borderRadius: '12px',
                              border: 'none',
                              boxShadow:
                                '0 10px 25px -5px rgb(15 23 42 / 0.25)',
                            }}
                          />
                          <Legend iconType="circle" />
                          {renderStrategyLines(primaryStrategy)}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </ChartCard>

                  {/* Waste */}
                  <ChartCard
                    title="Waste Accumulation"
                    badge="Total waste bags per day"
                    badgeColor="rose"
                    onClick={() => setFocusedChart('waste')}
                  >
                    <div className="h-[420px] w-full">
                      <ResponsiveContainer>
                        <LineChart
                          data={filteredSeriesData('total_waste_bags')}
                          margin={{
                            top: 10,
                            right: 30,
                            left: 10,
                            bottom: 5,
                          }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            vertical={false}
                            stroke="#e2e8f0"
                          />
                          <XAxis
                            dataKey="day"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#94a3b8' }}
                          />
                          <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#94a3b8' }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'rgba(255,255,255,0.96)',
                              borderRadius: '12px',
                              border: 'none',
                              boxShadow:
                                '0 10px 25px -5px rgb(15 23 42 / 0.25)',
                            }}
                          />
                          <Legend iconType="circle" />
                          {renderStrategyLines(primaryStrategy)}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </ChartCard>
                </div>

                <InsightsPanel
                  insights={insights}
                  primaryStrategy={primaryStrategy}
                />
              </div>

              {/* Bar comparison */}
              <div className="bg-white/95 backdrop-blur-lg p-8 rounded-2xl shadow-xl border border-slate-100 transition-all duration-300 hover:shadow-2xl hover:-translate-y-0.5 text-slate-800">
                <div className="mb-6 flex flex-col md:flex-row md:items-end md:justify-between gap-3">
                  <div>
                    <h3 className="text-xl font-bold text-slate-800 mb-1">
                      Overall Performance Comparison
                    </h3>
                    <p className="text-slate-500 text-sm">
                      Aggregated metrics averaged over the simulation period.
                      Higher bars are better for revenue; lower bars are better
                      for waste.
                    </p>
                  </div>
                  <p className="text-xs text-slate-400">
                    Highlighted strategy:{' '}
                    <span className="font-semibold text-slate-700">
                      {primaryStrategy}
                    </span>
                  </p>
                </div>
                <div className="h-[420px] w-full">
                  <ResponsiveContainer>
                    <BarChart
                      data={formatComparisonData(results)}
                      barSize={40}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      role="img"
                      aria-label="Bar chart comparing average revenue and average waste bags per strategy"
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        vertical={false}
                        stroke="#e2e8f0"
                      />
                      <XAxis
                        dataKey="name"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#64748b', fontWeight: 600 }}
                      />
                      <YAxis
                        yAxisId="left"
                        orientation="left"
                        stroke="#22c55e"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#94a3b8' }}
                        tickFormatter={(val) => `$${val}`}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        stroke="#ef4444"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#94a3b8' }}
                      />
                      <Tooltip
                        cursor={{ fill: '#f8fafc' }}
                        contentStyle={{
                          borderRadius: '12px',
                          border: 'none',
                          boxShadow:
                            '0 10px 25px -5px rgb(15 23 42 / 0.25)',
                          backgroundColor: 'rgba(255,255,255,0.96)',
                        }}
                      />
                      <Legend
                        wrapperStyle={{
                          paddingTop: '20px',
                          color: '#64748b',
                        }}
                      />
                      <Bar
                        yAxisId="left"
                        dataKey="revenue"
                        fill="url(#colorRevenue)"
                        name="Avg Revenue ($)"
                        radius={[6, 6, 0, 0]}
                      />
                      <Bar
                        yAxisId="right"
                        dataKey="waste"
                        fill="url(#colorWaste)"
                        name="Avg Waste Bags"
                        radius={[6, 6, 0, 0]}
                      />
                      <defs>
                        <linearGradient
                          id="colorRevenue"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="0%"
                            stopColor="#22c55e"
                            stopOpacity={0.95}
                          />
                          <stop
                            offset="100%"
                            stopColor="#16a34a"
                            stopOpacity={0.85}
                          />
                        </linearGradient>
                        <linearGradient
                          id="colorWaste"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="0%"
                            stopColor="#f97373"
                            stopOpacity={0.95}
                          />
                          <stop
                            offset="100%"
                            stopColor="#ef4444"
                            stopOpacity={0.85}
                          />
                        </linearGradient>
                      </defs>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Detailed table */}
              <div className="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
                <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/70">
                  <h3 className="text-lg font-bold text-slate-800">
                    Detailed Metrics Analysis
                  </h3>
                  <button className="text-sm text-emerald-600 font-medium hover:underline">
                    Download CSV
                  </button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs text-slate-500 uppercase bg-slate-50">
                      <tr>
                        <th className="px-6 py-4 font-semibold tracking-wider">
                          Metric
                        </th>
                        <th className="px-6 py-4 font-semibold tracking-wider">
                          Greedy
                        </th>
                        <th className="px-6 py-4 font-semibold tracking-wider text-blue-600">
                          Near-Opt
                        </th>
                        <th className="px-6 py-4 font-semibold tracking-wider">
                          RWES_T
                        </th>
                        <th className="px-6 py-4 font-semibold tracking-wider">
                          Yomna
                        </th>
                        <th className="px-6 py-4 font-semibold tracking-wider text-emerald-600">
                          Hybrid
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 bg-white">
                      {results.comparison.metric.map((metric, idx) => (
                        <tr
                          key={idx}
                          className="hover:bg-slate-50 transition-colors"
                        >
                          <td className="px-6 py-4 font-medium text-slate-700">
                            {metric}
                          </td>
                          <td className="px-6 py-4 text-slate-500">
                            {formatVal(
                              metric,
                              results.comparison.greedy[idx]
                            )}
                          </td>
                          <td className="px-6 py-4 font-medium text-blue-600 bg-blue-50/40">
                            {formatVal(
                              metric,
                              results.comparison.near_optimal[idx]
                            )}
                          </td>
                          <td className="px-6 py-4 text-slate-500">
                            {formatVal(
                              metric,
                              results.comparison.rwes_t[idx]
                            )}
                          </td>
                          <td className="px-6 py-4 text-slate-500">
                            {formatVal(
                              metric,
                              results.comparison.yomna[idx]
                            )}
                          </td>
                          <td className="px-6 py-4 font-medium text-emerald-600 bg-emerald-50/40">
                            {formatVal(
                              metric,
                              results.comparison.hybrid_enhanced[idx]
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Focused chart modal */}
          <ChartModal
            open={!!focusedChart}
            onClose={() => setFocusedChart(null)}
            type={focusedChart === 'waste' ? 'waste' : 'revenue'}
            data={
              focusedChart === 'waste'
                ? filteredSeriesData('total_waste_bags')
                : filteredSeriesData('total_revenue')
            }
            primaryStrategy={primaryStrategy}
          />
        </main>
      </div>
    </div>
  );
}

/* ------------ Small reusable pieces ------------ */

const ConfigField = ({ label, helper, children }) => (
  <div className="space-y-2">
    <label className="block text-xs font-medium uppercase tracking-wide text-slate-400">
      {label}
    </label>
    {children}
    <p className="text-[11px] text-slate-500">{helper}</p>
  </div>
);

const UploadCard = ({ label, subtitle, onChange, filename, accent }) => {
  const accentBorder =
    accent === 'emerald'
      ? 'hover:border-emerald-400/80'
      : 'hover:border-sky-400/80';
  const accentIcon =
    accent === 'emerald'
      ? 'group-hover:text-emerald-300'
      : 'group-hover:text-sky-300';

  return (
    <div className="flex-1">
      <label
        className={`flex flex-col items-center justify-center gap-2 p-5 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 cursor-pointer ${accentBorder} hover:bg-slate-950 transition-all group relative overflow-hidden`}
      >
        <Upload
          size={22}
          className={`text-slate-500 ${accentIcon} transition-colors`}
        />
        <span className="text-xs font-medium text-slate-200 group-hover:text-slate-50">
          {label}
        </span>
        <span className="text-[11px] text-slate-500 group-hover:text-slate-400">
          {subtitle}
        </span>
        <input
          type="file"
          className="hidden"
          onChange={onChange}
          accept=".csv"
        />
        {filename && (
          <div className="absolute top-2 right-2 bg-emerald-100 text-emerald-700 p-1 rounded-full">
            <CheckCircle2 size={16} />
          </div>
        )}
      </label>
      {filename && (
        <span className="block text-center mt-2 text-[11px] font-medium text-emerald-300">
          Loaded: {filename.slice(0, 24)}…
        </span>
      )}
    </div>
  );
};

const ScenarioSnapshot = ({ config }) => (
  <div className="bg-slate-900/80 rounded-2xl border border-slate-800 shadow-xl shadow-black/40 p-5 backdrop-blur transition-all duration-300 hover:shadow-emerald-500/20 hover:-translate-y-0.5">
    <h3 className="text-sm font-semibold text-slate-100 mb-3">
      Scenario snapshot
    </h3>
    <div className="grid grid-cols-3 gap-3 text-xs">
      <SnapshotCard label="Days simulated" value={config.num_days} />
      <SnapshotCard
        label="Stores"
        value={
          config.use_uploaded_data && config.stores_filename
            ? 'From CSV'
            : config.num_stores
        }
      />
      <SnapshotCard
        label="Customers"
        value={
          config.use_uploaded_data && config.customers_filename
            ? 'From CSV'
            : config.num_customers
        }
      />
    </div>
    <p className="mt-3 text-[11px] text-slate-500">
      Use this card in your report to concisely describe each experiment
      setup.
    </p>
  </div>
);

const SnapshotCard = ({ label, value }) => (
  <div className="rounded-xl bg-slate-950/70 border border-slate-800 px-3 py-2.5">
    <p className="text-slate-400 mb-1">{label}</p>
    <p className="text-lg font-semibold text-slate-50">{value}</p>
  </div>
);

const ControlPanel = ({
  primaryStrategy,
  setPrimaryStrategy,
  timeWindow,
  setTimeWindow,
}) => (
  <div className="bg-slate-900/80 rounded-2xl border border-slate-800 shadow-xl shadow-black/40 p-5 backdrop-blur space-y-4">
    <h3 className="text-sm font-semibold text-slate-100">View controls</h3>

    {/* Primary strategy */}
    <div className="space-y-1 text-xs">
      <p className="text-slate-400 mb-1">
        Highlight a strategy across all charts:
      </p>
      <div className="relative inline-block w-full">
        <select
          className="w-full appearance-none rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-2 text-xs font-medium text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/60 focus:border-emerald-400 pr-8"
          value={primaryStrategy}
          onChange={(e) => setPrimaryStrategy(e.target.value)}
        >
          {STRATEGIES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
      </div>
    </div>

    {/* Time window */}
    <div className="space-y-1 text-xs">
      <p className="text-slate-400 mb-1">Time window for daily charts:</p>
      <div className="inline-flex rounded-full bg-slate-950/70 border border-slate-800 p-1 text-[11px]">
        {[
          { id: 'all', label: 'All days' },
          { id: 'first_half', label: 'Start' },
          { id: 'second_half', label: 'End' },
        ].map((opt) => (
          <button
            key={opt.id}
            onClick={() => setTimeWindow(opt.id)}
            className={`px-3 py-1 rounded-full transition-colors ${
              timeWindow === opt.id
                ? 'bg-emerald-500 text-slate-950'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>

    <p className="text-[11px] text-slate-500">
      These controls only affect the visualisation layer; underlying
      simulation results remain unchanged, which you can reference in the
      numeric table.
    </p>
  </div>
);

/* Keyboard‑accessible chart card */
const ChartCard = ({ title, badge, badgeColor, onClick, children }) => {
  const badgeBg =
    badgeColor === 'emerald'
      ? 'bg-emerald-100 text-emerald-700'
      : 'bg-rose-100 text-rose-700';

  const handleKeyDown = (e) => {
    if (!onClick) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick();
    }
  };

  const clickableProps = onClick
    ? {
        role: 'button',
        tabIndex: 0,
        'aria-label': `Expand ${title} chart`,
        onKeyDown: handleKeyDown,
      }
    : {};

  return (
    <div
      {...clickableProps}
      onClick={onClick}
      className={`bg-white/90 backdrop-blur-md p-6 rounded-2xl shadow-xl border border-slate-100 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-2xl focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/70 ${
        onClick ? 'cursor-zoom-in' : ''
      }`}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-slate-800">{title}</h3>
        <span
          className={`text-xs font-medium px-2 py-1 rounded-full ${badgeBg}`}
        >
          {badge}
        </span>
      </div>
      {children}
    </div>
  );
};

/* Insights panel */
const InsightsPanel = ({ insights, primaryStrategy }) => (
  <div className="bg-slate-900/85 rounded-2xl border border-slate-800 shadow-2xl shadow-black/40 p-6 backdrop-blur flex flex-col gap-4 transition-all duration-300 hover:shadow-emerald-500/30 hover:-translate-y-0.5">
    <h3 className="text-lg font-semibold text-slate-50 flex items-center gap-2">
      <AlertCircle className="h-5 w-5 text-amber-300" />
      Key insights from this run
    </h3>
    {insights ? (
      <>
        <ul className="space-y-2 text-sm text-slate-200">
          <li>
            <span className="font-semibold text-emerald-300">
              Revenue:
            </span>{' '}
            {insights.bestRevenue.name} achieved the highest average daily
            revenue ({`$${insights.bestRevenue.value.toFixed(2)}`}),
            approximately {insights.revenueLiftVsGreedy.toFixed(1)}% higher
            than the Greedy baseline.
          </li>
          <li>
            <span className="font-semibold text-rose-300">Waste:</span>{' '}
            {insights.bestWaste.name} produced the lowest average waste (
            {insights.bestWaste.value.toFixed(
              1
            )}{' '}
            bags per day), about {insights.wasteDropVsGreedy.toFixed(1)}% lower
            than Greedy.
          </li>
          <li>
            <span className="font-semibold text-sky-300">
              Trade‑off:
            </span>{' '}
            {insights.tradeoffText}
          </li>
          <li>
            <span className="font-semibold text-violet-300">
              Focused view:
            </span>{' '}
            Currently highlighting{' '}
            <span className="font-semibold">{primaryStrategy}</span>; use this
            to guide the discussion on both charts.
          </li>
        </ul>
        <p className="text-[11px] text-slate-500 border-t border-slate-800 pt-3">
          Use these sentences directly in your report to justify why a specific
          policy outperforms the Greedy baseline and how the revenue–waste
          trade‑off behaves.
        </p>
      </>
    ) : (
      <p className="text-sm text-slate-400">
        Run a simulation to see automatically generated narrative insights.
      </p>
    )}
  </div>
);

/* Accessible modal */
const ChartModal = ({ open, onClose, type, data, primaryStrategy }) => {
  const closeBtnRef = useRef(null);

  useEffect(() => {
    if (open && closeBtnRef.current) {
      closeBtnRef.current.focus();
    }
  }, [open]);

  if (!open) return null;

  const isRevenue = type === 'revenue';
  const title = isRevenue
    ? 'Revenue Trends (focus view)'
    : 'Waste Accumulation (focus view)';
  const badge = isRevenue ? 'Total revenue per day' : 'Total waste bags per day';

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/70 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="chart-modal-title"
      aria-describedby="chart-modal-desc"
    >
      <div className="absolute inset-0" onClick={onClose} />
      <div className="relative z-50 w-full max-w-5xl mx-4 bg-slate-950/95 border border-slate-800 rounded-3xl shadow-2xl overflow-hidden transform transition-all duration-200 scale-100">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/80">
          <div>
            <h2
              id="chart-modal-title"
              className="text-lg font-semibold text-slate-50"
            >
              {title}
            </h2>
            <p
              id="chart-modal-desc"
              className="text-xs text-slate-400 mt-1"
            >
              Click outside the panel or press Escape to return to the
              dashboard.
            </p>
          </div>
          <button
            ref={closeBtnRef}
            onClick={onClose}
            className="rounded-full p-1.5 hover:bg-slate-800 text-slate-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
            aria-label="Close chart"
            type="button"
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-6 pt-4 flex items-center justify-between">
          <span className="text-xs font-medium px-2 py-1 rounded-full bg-slate-800 text-slate-200 border border-slate-700">
            {badge}
          </span>
          <span className="text-[11px] text-slate-400">
            Highlighted strategy:{' '}
            <span className="font-semibold text-emerald-300">
              {primaryStrategy}
            </span>
          </span>
        </div>

        <div className="h-[520px] w-full px-4 pb-6 pt-2">
          <ResponsiveContainer>
            <LineChart
              data={data}
              margin={{ top: 20, right: 40, left: 20, bottom: 20 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                stroke="#1e293b"
              />
              <XAxis
                dataKey="day"
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#9ca3af' }}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#9ca3af' }}
                tickFormatter={(val) =>
                  isRevenue ? `$${val}` : `${val}`
                }
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15,23,42,0.96)',
                  borderRadius: '12px',
                  border: '1px solid rgba(148,163,184,0.4)',
                  boxShadow: '0 18px 45px -15px rgba(15,23,42,0.9)',
                  color: '#e5e7eb',
                }}
              />
              <Legend iconType="circle" />
              {renderStrategyLines(primaryStrategy)}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

/* ------------ Toast ------------ */

const Toast = ({ message, type, onClose }) => {
  const bgColors = {
    success: 'bg-emerald-50 border-emerald-200 text-emerald-800',
    error: 'bg-rose-50 border-rose-200 text-rose-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800',
  };

  const icons = {
    success: <CheckCircle2 size={18} className="text-emerald-500" />,
    error: <AlertCircle size={18} className="text-rose-500" />,
    info: <Activity size={18} className="text-blue-500" />,
  };

  return (
    <div
      className={`flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg animate-in fade-in slide-in-from-right-8 ${
        bgColors[type]
      } min-w-[300px]`}
    >
      {icons[type]}
      <span className="text-sm font-medium flex-1">{message}</span>
      <button onClick={onClose} className="hover:opacity-70">
        <X size={16} />
      </button>
    </div>
  );
};

/* ------------ Metrics & helpers ------------ */

const KPICard = ({ title, value, icon, gradient, glow }) => (
  <div
    className={`bg-gradient-to-br ${gradient} p-6 rounded-2xl shadow-xl ${glow} text-white flex items-center justify-between transition-transform duration-200 hover:scale-[1.02] focus-within:ring-2 focus-within:ring-white/70`}
    role="group"
    aria-label={title}
  >
    <div>
      <div className="text-white/80 font-medium mb-1 text-sm">
        {title}
      </div>
      <div className="text-3xl font-bold tracking-tight">{value}</div>
    </div>
    <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm shadow-inner">
      {icon}
    </div>
  </div>
);

/* Case‑insensitive metric lookup */
const findMetricIndex = (comp, keyword) =>
  comp.metric.findIndex((m) =>
    m.toLowerCase().includes(keyword.toLowerCase())
  );

const getMinMetric = (comp, keyword) => {
  const idx = findMetricIndex(comp, keyword);
  if (idx === -1) return '-';
  const vals = [
    comp.greedy[idx],
    comp.near_optimal[idx],
    comp.rwes_t[idx],
    comp.yomna[idx],
    comp.hybrid_enhanced[idx],
  ];
  return Math.min(...vals).toFixed(1);
};

const getMaxMetric = (comp, keyword, isMoney = true) => {
  const idx = findMetricIndex(comp, keyword);
  if (idx === -1) return '-';
  const vals = [
    comp.greedy[idx],
    comp.near_optimal[idx],
    comp.rwes_t[idx],
    comp.yomna[idx],
    comp.hybrid_enhanced[idx],
  ];
  const maxVal = Math.max(...vals);
  return isMoney ? `$${maxVal.toFixed(2)}` : maxVal.toFixed(2);
};

const getBestStrategyByMetric = (comp, keyword, type = 'max') => {
  const idx = findMetricIndex(comp, keyword);
  if (idx === -1) return null;

  const entries = [
    ['Greedy', comp.greedy[idx]],
    ['Near-Opt', comp.near_optimal[idx]],
    ['RWES_T', comp.rwes_t[idx]],
    ['Yomna', comp.yomna[idx]],
    ['Hybrid', comp.hybrid_enhanced[idx]],
  ];
  const sorted = entries.sort((a, b) =>
    type === 'max' ? b[1] - a[1] : a[1] - b[1]
  );
  const [name, value] = sorted[0];
  return { name, value };
};

const deriveInsights = (comp, primaryStrategy) => {
  const bestRevenue = getBestStrategyByMetric(
    comp,
    'Revenue per Day',
    'max'
  );
  const bestWaste = getBestStrategyByMetric(
    comp,
    'Waste Bags per Day',
    'min'
  );
  if (!bestRevenue || !bestWaste) return null;

  const revIdx = findMetricIndex(comp, 'Revenue per Day');
  const wasteIdx = findMetricIndex(comp, 'Waste Bags per Day');

  const greedyRev = comp.greedy[revIdx];
  const greedyWaste = comp.greedy[wasteIdx];

  const revenueLiftVsGreedy =
    greedyRev > 0 ? ((bestRevenue.value - greedyRev) / greedyRev) * 100 : 0;
  const wasteDropVsGreedy =
    greedyWaste > 0 ? ((greedyWaste - bestWaste.value) / greedyWaste) * 100 : 0;

  const tradeoffText =
    bestRevenue.name === bestWaste.name
      ? `${bestRevenue.name} simultaneously maximizes revenue and minimizes waste, making it a strong candidate as the primary policy.`
      : `${bestRevenue.name} is best for revenue, while ${bestWaste.name} is best for waste. This highlights a classic efficiency vs. sustainability trade‑off to discuss in your analysis.`;

  return {
    bestRevenue,
    bestWaste,
    revenueLiftVsGreedy,
    wasteDropVsGreedy,
    tradeoffText,
    primaryStrategy,
  };
};

const formatComparisonData = (results) => {
  const comp = results.comparison;
  const revIdx = findMetricIndex(comp, 'Revenue per Day');
  const wasteIdx = findMetricIndex(comp, 'Waste Bags per Day');
  if (revIdx === -1 || wasteIdx === -1) return [];

  return [
    {
      name: 'Greedy',
      revenue: comp.greedy[revIdx],
      waste: comp.greedy[wasteIdx],
    },
    {
      name: 'Near-Opt',
      revenue: comp.near_optimal[revIdx],
      waste: comp.near_optimal[wasteIdx],
    },
    {
      name: 'RWES_T',
      revenue: comp.rwes_t[revIdx],
      waste: comp.rwes_t[wasteIdx],
    },
    {
      name: 'Yomna',
      revenue: comp.yomna[revIdx],
      waste: comp.yomna[wasteIdx],
    },
    {
      name: 'Hybrid',
      revenue: comp.hybrid_enhanced[revIdx],
      waste: comp.hybrid_enhanced[wasteIdx],
    },
  ];
};

const keyIsRatio = (k) => k.includes('Rate') || k.includes('%');

const formatVal = (metric, val) => {
  if (metric.includes('Revenue') || metric.includes('$')) {
    return `$${val.toFixed(2)}`;
  }
  if (metric.includes('Rate') || metric.includes('%') || keyIsRatio(metric)) {
    return `${val.toFixed(1)}%`;
  }
  return val.toFixed(1);
};

const prepareMultiStrategyData = (results, metricKey) => {
  if (!results?.near_optimal_results) return [];

  const days = results.near_optimal_results.daily_kpis.length;
  const data = [];

  for (let i = 0; i < days; i++) {
    const dayData = { day: i + 1 };

    if (results.greedy_results) {
      dayData['Greedy'] = results.greedy_results.daily_kpis[i][metricKey];
    }
    if (results.near_optimal_results) {
      dayData['Near-Opt'] =
        results.near_optimal_results.daily_kpis[i][metricKey];
    }
    if (results.rwes_t_results) {
      dayData['RWES_T'] = results.rwes_t_results.daily_kpis[i][metricKey];
    }
    if (results.yomna_results) {
      dayData['Yomna'] = results.yomna_results.daily_kpis[i][metricKey];
    }
    if (results.hybrid_enhanced_results) {
      dayData['Hybrid'] =
        results.hybrid_enhanced_results.daily_kpis[i][metricKey];
    }
    if (results.anan_results) {
      dayData['Anan'] = results.anan_results.daily_kpis[i][metricKey];
    }

    data.push(dayData);
  }
  return data;
};

/* Highlighted vs normal strategy lines */
const renderStrategyLines = (primary) => {
  const baseProps = (name, color) => ({
    type: 'monotone',
    dataKey: name,
    stroke: color,
    dot: false,
  });

  const defs = [
    ['Greedy', '#94a3b8', '5 5'],
    ['Near-Opt', '#3b82f6'],
    ['RWES_T', '#8b5cf6'],
    ['Yomna', '#ec4899'],
    ['Hybrid', '#10b981'],
  ];

  return defs.map(([name, color, dash]) => {
    const isPrimary = name === primary;
    return (
      <Line
        key={name}
        {...baseProps(name, color)}
        strokeWidth={isPrimary ? 4 : 2}
        strokeDasharray={name === 'Greedy' ? '5 5' : dash}
        activeDot={isPrimary ? { r: 4 } : false}
        opacity={isPrimary ? 1 : 0.6}
      />
    );
  });
};

/* Filter time window */
const filterByWindow = (data, window) => {
  if (!data || data.length === 0) return data;
  if (window === 'all') return data;

  const mid = Math.floor(data.length / 2);
  if (window === 'first_half') return data.slice(0, mid || 1);
  if (window === 'second_half') return data.slice(mid);
  return data;
};

export default App;
