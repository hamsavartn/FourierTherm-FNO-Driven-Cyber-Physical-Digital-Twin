"use client"

import React, { useState, useEffect, Suspense } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  ReferenceLine,
  Label,
  Legend,
} from 'recharts'
import { Activity, Thermometer, Droplets, Zap, AlertTriangle, RefreshCw, Settings2, Cable } from 'lucide-react'

// Lazy load Spline to prevent SSR issues and use Suspense for loading state
const Spline = React.lazy(() => import('@splinetool/react-spline'))

function FluidPreloader({ onComplete }: { onComplete: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onComplete, 3000)
    return () => clearTimeout(timer)
  }, [onComplete])

  return (
    <motion.div 
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, filter: "blur(20px)" }}
      transition={{ duration: 1, ease: "easeInOut" }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black overflow-hidden"
    >
      {/* Fluid Orbs */}
      <motion.div 
        animate={{ 
          scale: [1, 1.5, 1],
          x: [0, 100, -100, 0],
          y: [0, -50, 50, 0],
          rotate: [0, 180, 360]
        }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        className="absolute w-96 h-96 bg-blue-600/30 rounded-full blur-[80px] mix-blend-screen"
      />
      <motion.div 
        animate={{ 
          scale: [1, 1.2, 1],
          x: [0, -150, 150, 0],
          y: [0, 100, -100, 0],
          rotate: [360, 180, 0]
        }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
        className="absolute w-96 h-96 bg-teal-500/30 rounded-full blur-[80px] mix-blend-screen"
      />
      <motion.div 
        animate={{ 
          scale: [1, 1.8, 1],
          x: [100, -100, 0],
          y: [-100, 100, 0]
        }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        className="absolute w-[500px] h-[500px] bg-primary/20 rounded-full blur-[100px] mix-blend-screen"
      />
      
      {/* Text Reveal */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.9, letterSpacing: "0px" }}
        animate={{ opacity: 1, scale: 1, letterSpacing: "8px" }}
        transition={{ duration: 2, ease: "easeOut" }}
        className="relative z-10 text-white font-black text-2xl uppercase tracking-[8px] text-center"
      >
        Initializing <br/>
        <span className="text-primary text-xl font-medium tracking-[4px]">Cyber-Physical Engine</span>
      </motion.div>
    </motion.div>
  )
}

export default function Dashboard() {
  const [showPreloader, setShowPreloader] = useState(true)
  const [data, setData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Interactive Simulation State
  const [days, setDays] = useState(1)
  const [pa, setPa] = useState(1000)
  const [pb, setPb] = useState(1000)
  const [pc, setPc] = useState(1000)
  
  // Stats
  const [maxTemp, setMaxTemp] = useState(0)
  const [minMoisture, setMinMoisture] = useState(0)

  // Debounced Fetch Logic - Reduced to 150ms for snappiness
  useEffect(() => {
    if (showPreloader) return // Don't fetch until preloader is done
    
    const handler = setTimeout(() => {
      fetchData()
    }, 150)

    return () => clearTimeout(handler)
  }, [days, pa, pb, pc, showPreloader])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('http://127.0.0.1:8000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days, pa, pb, pc })
      })
      
      if (!res.ok) throw new Error("Failed to fetch simulation data")
        
      const json = await res.json()
      
      // Transform data for Recharts
      const transformed = json.time_series.map((pt: any) => ({
        time: `${(pt.time / 3600).toFixed(1)}h`,
        coreA: pt.CoreA,
        coreB: pt.CoreB,
        coreC: pt.CoreC,
        sheath: pt.Sheath,
        soil: pt.Soil,
        moisture: pt.Moisture
      }))
      
      setData(transformed)
      setMaxTemp(Math.max(json.max_temps.CoreA, json.max_temps.CoreB, json.max_temps.CoreC))
      setMinMoisture(Math.min(...json.time_series.map((pt: any) => pt.Moisture)))
      
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-black text-white selection:bg-primary/30 font-sans">
      
      <AnimatePresence>
        {showPreloader && <FluidPreloader onComplete={() => setShowPreloader(false)} />}
      </AnimatePresence>

      {/* 3D Environment with Suspense */}
      <div className="absolute inset-0 z-0 opacity-50 pointer-events-none">
        <Suspense fallback={
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-32 h-32 border-4 border-primary/20 border-t-primary rounded-full animate-spin" />
          </div>
        }>
          {/* Using a reliable abstract Spline scene */}
          <Spline scene="https://prod.spline.design/6Wq1Q7YGyM-iab9i/scene.splinecode" />
        </Suspense>
        {/* CSS Fallback Glows just in case */}
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[120px] mix-blend-screen" />
        <div className="absolute top-[40%] -right-[10%] w-[40%] h-[60%] rounded-full bg-blue-600/10 blur-[120px] mix-blend-screen" />
      </div>

      {!showPreloader && (
        <main className="relative z-10 container mx-auto max-w-7xl px-4 py-8">
          
          {/* Header */}
          <motion.header 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4"
          >
            <div>
              <div className="flex items-center gap-3 mb-3">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Badge variant="glass" className="px-3 py-1.5 cursor-pointer text-primary shadow-[0_0_15px_rgba(var(--primary),0.3)]">
                    <Zap className="w-3.5 h-3.5 mr-1" /> Live Digital Twin
                  </Badge>
                </motion.div>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Badge variant="outline" className="px-3 py-1.5 cursor-pointer border-white/20 text-white/80 hover:bg-white/10 transition-colors">
                    <Cable className="w-3.5 h-3.5 mr-1" /> 11kV XLPE Cable
                  </Badge>
                </motion.div>
              </div>
              <h1 className="text-4xl md:text-6xl font-extrabold tracking-tighter bg-clip-text text-transparent bg-gradient-to-br from-white via-white/90 to-white/40">
                FourierTherm
              </h1>
            </div>
          </motion.header>

          {/* Bento Box Grid */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            
            {/* Controls Panel (Left Col) */}
            <motion.div 
              className="md:col-span-4 lg:col-span-3 flex flex-col gap-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <Card className="flex-1 bg-black/40 backdrop-blur-xl border-white/10 shadow-2xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings2 className="w-5 h-5 text-primary" /> Parameters
                  </CardTitle>
                  <CardDescription>Adjust load dynamically</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-white/70">Timeline (Days)</span>
                      <span className="font-mono text-primary">{days} d</span>
                    </div>
                    <Slider 
                      min={1} max={7} step={1} value={days} 
                      onChange={(e) => setDays(Number(e.target.value))} 
                    />
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-white/70">Phase A Current</span>
                      <span className="font-mono text-orange-400">{pa} A</span>
                    </div>
                    <Slider 
                      min={400} max={2000} step={50} value={pa} 
                      onChange={(e) => setPa(Number(e.target.value))} 
                    />
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-white/70">Phase B Current</span>
                      <span className="font-mono text-blue-400">{pb} A</span>
                    </div>
                    <Slider 
                      min={400} max={2000} step={50} value={pb} 
                      onChange={(e) => setPb(Number(e.target.value))} 
                    />
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-white/70">Phase C Current</span>
                      <span className="font-mono text-purple-400">{pc} A</span>
                    </div>
                    <Slider 
                      min={400} max={2000} step={50} value={pc} 
                      onChange={(e) => setPc(Number(e.target.value))} 
                    />
                  </div>

                </CardContent>
              </Card>
            </motion.div>

            {/* Main Visualization (Right Col) */}
            <div className="md:col-span-8 lg:col-span-9 flex flex-col gap-6">
              
              {/* Stats Row */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                <StatCard 
                  title="Peak Core Temp" 
                  value={loading ? "-" : `${maxTemp.toFixed(1)}°C`}
                  icon={<Thermometer className="w-5 h-5 text-red-400" />}
                  trend={maxTemp > 90 ? "Critical" : "Stable"}
                  trendColor={maxTemp > 90 ? "text-red-400 border-red-500/30" : "text-green-400 border-green-500/30"}
                  delay={0.3}
                />
                <StatCard 
                  title="Min Soil Moisture" 
                  value={loading ? "-" : minMoisture.toFixed(3)}
                  icon={<Droplets className="w-5 h-5 text-blue-400" />}
                  trend={minMoisture < 0.05 ? "Dryout Risk" : "Stable"}
                  trendColor={minMoisture < 0.05 ? "text-red-400 border-red-500/30" : "text-blue-400 border-blue-500/30"}
                  delay={0.4}
                />
                <StatCard 
                  title="System Status" 
                  value={loading ? "Computing" : (maxTemp > 90 || minMoisture < 0.05) ? "Warning" : "Optimal"}
                  icon={<Activity className="w-5 h-5 text-emerald-400" />}
                  trend={loading ? "RK4 Solver" : "Synced"}
                  trendColor="text-white/50 border-white/10"
                  delay={0.5}
                />
              </div>

              {/* Charts Row */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.6 }}
                  className="h-full"
                >
                  <Card className="h-full min-h-[350px] bg-black/40 backdrop-blur-xl border-white/10 shadow-2xl flex flex-col">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <Thermometer className="w-5 h-5 text-red-400" /> Temperature Dynamics
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 relative">
                      <AnimatePresence>
                        {loading && (
                          <motion.div 
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-10 rounded-xl"
                          >
                            <RefreshCw className="w-8 h-8 text-primary animate-spin" />
                          </motion.div>
                        )}
                      </AnimatePresence>
                      {error ? (
                        <div className="w-full h-full flex items-center justify-center text-red-400">
                          <AlertTriangle className="w-6 h-6 mr-2" /> {error}
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                            <XAxis dataKey="time" stroke="rgba(255,255,255,0.3)" fontSize={11} tickLine={false} axisLine={false} />
                            <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} tickLine={false} axisLine={false} />
                            <Tooltip 
                              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', backdropFilter: 'blur(10px)' }}
                              itemStyle={{ color: '#fff' }}
                              animationDuration={300}
                            />
                            <ReferenceLine y={90} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.8}>
                              <Label value="Critical Limit (90°C)" position="insideTopLeft" fill="#ef4444" fontSize={11} opacity={0.8} />
                            </ReferenceLine>
                            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px', color: 'rgba(255,255,255,0.7)' }} />
                            <Line type="monotone" dataKey="coreA" name="Core A °C" stroke="#ef4444" strokeWidth={2} dot={false} activeDot={{ r: 6, fill: '#ef4444' }} />
                            <Line type="monotone" dataKey="coreB" name="Core B °C" stroke="#f43f5e" strokeWidth={2} dot={false} activeDot={{ r: 6, fill: '#f43f5e' }} />
                            <Line type="monotone" dataKey="coreC" name="Core C °C" stroke="#fb7185" strokeWidth={2} dot={false} activeDot={{ r: 6, fill: '#fb7185' }} />
                            <Line type="monotone" dataKey="sheath" name="Sheath °C" stroke="#f97316" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="soil" name="Soil °C" stroke="#eab308" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.7 }}
                  className="h-full"
                >
                  <Card className="h-full min-h-[350px] bg-black/40 backdrop-blur-xl border-white/10 shadow-2xl flex flex-col">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <Droplets className="w-5 h-5 text-blue-400" /> Moisture Migration
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 relative">
                      <AnimatePresence>
                        {loading && (
                          <motion.div 
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-10 rounded-xl"
                          >
                            <RefreshCw className="w-8 h-8 text-primary animate-spin" />
                          </motion.div>
                        )}
                      </AnimatePresence>
                      {error ? (
                        <div className="w-full h-full flex items-center justify-center text-red-400">
                          <AlertTriangle className="w-6 h-6 mr-2" /> {error}
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                            <defs>
                              <linearGradient id="colorMoisture" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                            <XAxis dataKey="time" stroke="rgba(255,255,255,0.3)" fontSize={11} tickLine={false} axisLine={false} />
                            <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                            <Tooltip 
                              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', backdropFilter: 'blur(10px)' }}
                              itemStyle={{ color: '#fff' }}
                              animationDuration={300}
                            />
                            <ReferenceLine y={0.05} stroke="#f87171" strokeDasharray="3 3" strokeOpacity={0.6}>
                              <Label value="Dryout Risk (<0.05)" position="insideBottomLeft" fill="#f87171" fontSize={11} opacity={0.8} />
                            </ReferenceLine>
                            <Area type="monotone" dataKey="moisture" name="Moisture" stroke="#3b82f6" fillOpacity={1} fill="url(#colorMoisture)" strokeWidth={2} />
                          </AreaChart>
                        </ResponsiveContainer>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              </div>
              
            </div>
          </div>
        </main>
      )}
    </div>
  )
}

function StatCard({ title, value, icon, trend, trendColor, delay }: { title: string, value: string, icon: React.ReactNode, trend: string, trendColor: string, delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay, ease: "easeOut" }}
      whileHover={{ y: -5 }}
    >
      <Card className="relative overflow-hidden group bg-black/40 backdrop-blur-xl border-white/10 shadow-lg hover:border-white/30 transition-all duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <CardContent className="p-6">
          <div className="flex justify-between items-start mb-6">
            <p className="text-sm font-medium text-white/60">{title}</p>
            <motion.div 
              whileHover={{ rotate: 15, scale: 1.1 }}
              className="p-2 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10"
            >
              {icon}
            </motion.div>
          </div>
          <div className="flex items-end justify-between">
            <h2 className="text-4xl font-black text-white tracking-tighter">{value}</h2>
            <span className={cn("text-xs font-semibold px-2.5 py-1 rounded-full bg-white/5 border backdrop-blur-md", trendColor)}>
              {trend}
            </span>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
