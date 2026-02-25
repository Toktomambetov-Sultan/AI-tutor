import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import './Login.css';

export default function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuthStore();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await login(email, password);
            navigate('/dashboard');
        } catch (err) {
            setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-page">
            <div className="auth-brand">
                <div className="auth-brand-content">
                    <div className="auth-brand-icon">🎓</div>
                    <h1>AI Teacher</h1>
                    <p>Your intelligent learning companion. Access courses, track progress, and get AI-powered tutoring — all in one place.</p>
                </div>
            </div>

            <div className="auth-form-panel">
                <div className="auth-form-container">
                    <h2>Welcome back</h2>
                    <p className="auth-subtitle">Sign in to your account to continue</p>

                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="email">Email</label>
                            <input
                                id="email"
                                type="email"
                                value={email}
                                onChange={e => setEmail(e.target.value)}
                                placeholder="you@example.com"
                                autoComplete="email"
                                required
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="password">Password</label>
                            <input
                                id="password"
                                type="password"
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                placeholder="Enter your password"
                                autoComplete="current-password"
                                required
                            />
                        </div>

                        {error && <div className="alert alert-error">{error}</div>}

                        <button type="submit" className="btn-primary btn-full" disabled={loading}>
                            {loading ? 'Signing in...' : 'Sign In'}
                        </button>
                    </form>

                    <p className="auth-link">
                        Student? <Link to="/register">Create an account</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
