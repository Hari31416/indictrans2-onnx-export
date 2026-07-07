import { useState } from 'react'
import { OverviewTab } from '@/components/OverviewTab'
import { BenchmarkTab } from '@/components/BenchmarkTab'
import { MismatchTab } from '@/components/MismatchTab'
import { BlogTab } from '@/components/BlogTab'
import { LiveBenchmarkTab } from '@/components/LiveBenchmarkTab'
import { LayoutDashboard, Activity, AlertTriangle, BookOpen, Volume2, Cpu, ShieldAlert, Wrench, Zap } from 'lucide-react'

const GithubIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth="2"
    fill="none"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
  </svg>
)

type TabType =
  | 'overview'
  | 'benchmarks'
  | 'mismatches'
  | 'live-benchmark'
  | 'doc-overview'
  | 'doc-architecture'
  | 'doc-export'
  | 'doc-quantization'
  | 'doc-optimization'

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('overview')

  const dashboardItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'benchmarks', label: 'Detailed Benchmarks', icon: Activity },
    { id: 'live-benchmark', label: 'Live Latency', icon: Cpu },
    { id: 'mismatches', label: 'Mismatch Explorer', icon: AlertTriangle }
  ] as const

  const docItems = [
    { id: 'doc-overview', label: 'Project Overview', icon: BookOpen },
    { id: 'doc-architecture', label: 'ONNX Graph Model', icon: Cpu },
    { id: 'doc-export', label: 'Export Journey', icon: Wrench },
    { id: 'doc-quantization', label: 'Quantization Logs', icon: ShieldAlert },
    { id: 'doc-optimization', label: 'Size & Optimization', icon: Zap }
  ] as const

  return (
    <div className="flex flex-col min-h-screen text-zinc-100 bg-transparent font-sans">
      {/* Top Header */}
      <header className="glass sticky top-0 z-50 flex items-center justify-between px-6 py-4 border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-teal-500 rounded-lg shadow text-zinc-950">
            <Volume2 size={18} />
          </div>
          <h1 className="text-base font-extrabold tracking-tight text-zinc-100">
            IndicTrans2 ONNX
          </h1>
        </div>
        <div className="flex items-center gap-4">
          <a
            href="https://github.com/Hari31416/indictrans2-onnx-export"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-400 hover:text-teal-400 transition"
          >
            <GithubIcon width={18} height={18} />
          </a>
        </div>
      </header>

      {/* Main Layout Container */}
      <div className="flex-1 flex flex-col md:flex-row">
        {/* Navigation Sidebar */}
        <aside className="w-full md:w-64 glass border-r border-white/5 p-4 flex flex-col gap-6 md:h-[calc(100vh-69px)] md:sticky md:top-[69px] overflow-y-auto">
          {/* Dashboard section */}
          <div className="space-y-1.5">
            <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-3 mb-2 hidden md:block">
              Dashboard
            </div>
            <nav className="flex flex-row md:flex-col gap-1 overflow-x-auto md:overflow-visible pb-2 md:pb-0">
              {dashboardItems.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-3 px-4 py-2.5 rounded-xl text-xs font-semibold whitespace-nowrap md:whitespace-normal transition-all duration-200 w-full ${isActive
                        ? 'bg-teal-500/10 border border-teal-500/20 text-teal-300 shadow'
                        : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-200 border border-transparent'
                      }`}
                  >
                    <Icon size={16} className={isActive ? 'text-teal-400' : 'text-zinc-400'} />
                    <span>{item.label}</span>
                  </button>
                )
              })}
            </nav>
          </div>

          {/* Technical Docs section */}
          <div className="space-y-1.5">
            <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-3 mb-2 hidden md:block">
              Technical Docs
            </div>
            <nav className="flex flex-row md:flex-col gap-1 overflow-x-auto md:overflow-visible pb-2 md:pb-0">
              {docItems.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-3 px-4 py-2.5 rounded-xl text-xs font-semibold whitespace-nowrap md:whitespace-normal transition-all duration-200 w-full ${isActive
                        ? 'bg-teal-500/10 border border-teal-500/20 text-teal-300 shadow'
                        : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-200 border border-transparent'
                      }`}
                  >
                    <Icon size={16} className={isActive ? 'text-teal-400' : 'text-zinc-400'} />
                    <span>{item.label}</span>
                  </button>
                )
              })}
            </nav>
          </div>
        </aside>

        {/* Core Tab Panels */}
        <main className="flex-1 p-6 md:p-8 overflow-y-auto max-w-7xl mx-auto w-full">
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'benchmarks' && <BenchmarkTab />}
          {activeTab === 'live-benchmark' && <LiveBenchmarkTab />}
          {activeTab === 'mismatches' && <MismatchTab />}
          {activeTab.startsWith('doc-') && <BlogTab activeSection={activeTab as any} />}
        </main>
      </div>
    </div>
  )
}
