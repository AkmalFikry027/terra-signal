import axios from 'axios'

export const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const KEY = 'satscope.token'
export const getToken = () => localStorage.getItem(KEY)
export const setToken = t => (t ? localStorage.setItem(KEY, t) : localStorage.removeItem(KEY))

export const api = axios.create({ baseURL: API, timeout: 60000 })
api.interceptors.request.use(cfg => {
  const t = getToken()
  if (t) cfg.headers.Authorization = `Bearer ${t}`
  return cfg
})

let onUnauthorized = () => {}
export const setUnauthorizedHandler = fn => { onUnauthorized = fn }
api.interceptors.response.use(r => r, err => {
  const url = err.config?.url || ''
  if (err.response?.status === 401 && !url.includes('/auth/login')) onUnauthorized()
  return Promise.reject(err)
})

export const errorText = (e, fallback = 'Something went wrong.') => {
  const d = e?.response?.data?.detail
  if (Array.isArray(d)) return d.map(x => x.msg).join(' ')
  if (typeof d === 'string') return d
  if (e?.code === 'ERR_NETWORK' || !e?.response) return 'Cannot reach the SAT-SCOPE API. Start the FastAPI backend on port 8000.'
  return fallback
}
