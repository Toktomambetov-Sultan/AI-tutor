import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
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
    const { t } = useTranslation();
    const { register } = useAuthStore();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        if (password !== confirmPassword) {
            setError(t('auth.passwordMismatch'));
            return;
        }
        if (password.length < 8) {
            setError(t('auth.passwordShort'));
            return;
        }

        setLoading(true);
        try {
            await register(fullName, email, password);
            setSuccess(true);
            setTimeout(() => navigate('/'), 2500);
        } catch (err) {
            setError(err.response?.data?.detail || t('auth.registrationFailed'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-page">
            <div className="auth-brand">
                <div className="auth-brand-content">
                    <div className="auth-brand-icon">🎓</div>
                    <h1>{t('auth.brandTitleRegister')}</h1>
                    <p>{t('auth.brandTextRegister')}</p>
                </div>
            </div>

            <div className="auth-form-panel">
                <div className="auth-form-container">
                    <h2>{t('auth.registerTitle')}</h2>
                    <p className="auth-subtitle">{t('auth.registerSubtitle')}</p>

                    {success ? (
                        <div className="success-box">
                            <span className="success-icon">✅</span>
                            <strong>{t('auth.registrationSuccess')}</strong>
                            <p style={{ marginTop: '0.5rem' }}>{t('auth.redirecting')}</p>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit}>
                            <div className="form-group">
                                <label htmlFor="fullName">{t('auth.fullName')}</label>
                                <input id="fullName" type="text" value={fullName} onChange={e => setFullName(e.target.value)} placeholder="John Doe" required />
                            </div>
                            <div className="form-group">
                                <label htmlFor="email">{t('auth.email')}</label>
                                <input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required />
                            </div>
                            <div className="form-group">
                                <label htmlFor="password">{t('auth.password')}</label>
                                <input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Minimum 8 characters" required />
                            </div>
                            <div className="form-group">
                                <label htmlFor="confirmPassword">{t('auth.confirmPassword')}</label>
                                <input id="confirmPassword" type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} placeholder="Repeat your password" required />
                            </div>

                            {error && <div className="alert alert-error">{error}</div>}

                            <button type="submit" className="btn-primary btn-full" disabled={loading}>
                                {loading ? t('auth.creatingAccount') : t('auth.createAccount')}
                            </button>
                        </form>
                    )}

                    <p className="auth-link">
                        {t('auth.alreadyHaveAccount')} <Link to="/">{t('auth.signInLink')}</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
