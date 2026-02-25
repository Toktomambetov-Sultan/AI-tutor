import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import './Login.css';

export default function Register() {
    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState(false);
    const [loading, setLoading] = useState(false);
    const { register } = useAuthStore();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }
        if (password.length < 8) {
            setError('Password must be at least 8 characters');
            return;
        }

        setLoading(true);
        try {
            await register(fullName, email, password);
            setSuccess(true);
            setTimeout(() => navigate('/'), 2500);
        } catch (err) {
            setError(err.response?.data?.detail || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-page">
            <div className="auth-brand">
                <div className="auth-brand-content">
                    <div className="auth-brand-icon">🎓</div>
                    <h1>Join AI Teacher</h1>
                    <p>Create a student account and start exploring courses from expert teachers. Learn at your own pace with AI support.</p>
                </div>
            </div>

            <div className="auth-form-panel">
                <div className="auth-form-container">
                    <h2>Create account</h2>
                    <p className="auth-subtitle">Register as a student to get started</p>

                    {success ? (
                        <div className="success-box">
                            <span className="success-icon">✅</span>
                            <strong>Registration successful!</strong>
                            <p style={{ marginTop: '0.5rem' }}>Redirecting to login...</p>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit}>
                            <div className="form-group">
                                <label htmlFor="fullName">Full Name</label>
                                <input id="fullName" type="text" value={fullName} onChange={e => setFullName(e.target.value)} placeholder="John Doe" required />
                            </div>
                            <div className="form-group">
                                <label htmlFor="email">Email</label>
                                <input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required />
                            </div>
                            <div className="form-group">
                                <label htmlFor="password">Password</label>
                                <input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Minimum 8 characters" required />
                            </div>
                            <div className="form-group">
                                <label htmlFor="confirmPassword">Confirm Password</label>
                                <input id="confirmPassword" type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} placeholder="Repeat your password" required />
                            </div>

                            {error && <div className="alert alert-error">{error}</div>}

                            <button type="submit" className="btn-primary btn-full" disabled={loading}>
                                {loading ? 'Creating account...' : 'Create Account'}
                            </button>
                        </form>
                    )}

                    <p className="auth-link">
                        Already have an account? <Link to="/">Sign in</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
