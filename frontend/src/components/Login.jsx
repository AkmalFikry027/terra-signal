import { useState } from 'react'
import { Satellite, Eye, EyeOff, LogIn, AlertTriangle } from 'lucide-react'
import { api, setToken, errorText } from '../lib/api'

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const submit = async e => {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      const { data } = await api.post('/api/auth/login', { username, password })
      setToken(data.access_token)
      onLogin(data.user)
    } catch (err) { setError(errorText(err, 'Sign-in failed.')) } finally { setBusy(false) }
  }

  return <div className="login-page">
    <form className="login-card" onSubmit={submit}>
      <div className="brand login-brand"><div className="brand-mark"><Satellite size={21} /></div><div><b>SAT<span>-</span>SCOPE</b><small>EARTH INTELLIGENCE</small></div></div>
      <h1>Sign in</h1>
      <p>NASA Space Apps · Be An Earth System Trend Detective</p>
      {error && <div className="form-alert"><AlertTriangle size={15} /> {error}</div>}
      <label>Username<input autoFocus autoComplete="username" value={username} onChange={e => setUsername(e.target.value)} required /></label>
      <label>Password
        <span className="pw-field">
          <input type={show ? 'text' : 'password'} autoComplete="current-password" value={password} onChange={e => setPassword(e.target.value)} required />
          <button type="button" onClick={() => setShow(s => !s)} aria-label="Toggle password visibility">{show ? <EyeOff size={16} /> : <Eye size={16} />}</button>
        </span>
      </label>
      <button className="primary-btn" disabled={busy}><LogIn size={16} /> {busy ? 'Signing in…' : 'Sign in'}</button>
      <small className="login-foot">© {new Date().getFullYear()} TerraSignal</small>
    </form>
  </div>
}
