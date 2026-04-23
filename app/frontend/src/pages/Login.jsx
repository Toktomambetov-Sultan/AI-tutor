import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../store/authStore';
import './Login.css';

export default function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { t } = useTranslation();
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
            setError(err.response?.data?.detail || t('auth.loginFailed'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-page">
            <div className="auth-brand">
                <div className="auth-brand-content">
                    <div className="auth-brand-icon">🎓</div>
                    <h1>{t('auth.brandTitleLogin')}</h1>
                    <p>{t('auth.brandTextLogin')}</p>
                </div>
            </div>

            <div className="auth-form-panel">
                <div className="auth-form-container">
                    <h2>{t('auth.welcomeBack')}</h2>
                    <p className="auth-subtitle">{t('auth.signInSubtitle')}</p>

                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="email">{t('auth.email')}</label>
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
                            <label htmlFor="password">{t('auth.password')}</label>
                            <input
                                id="password"
                                type="password"
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                placeholder={t('auth.password')}
                                autoComplete="current-password"
                                required
                            />
                        </div>

                        {error && <div className="alert alert-error">{error}</div>}

                        <button type="submit" className="btn-primary btn-full" disabled={loading}>
                            {loading ? t('auth.signingIn') : t('auth.signIn')}
                        </button>
                    </form>

                    <p className="auth-link">
                        {t('auth.studentQuestion')} <Link to="/register">{t('auth.createAnAccount')}</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
