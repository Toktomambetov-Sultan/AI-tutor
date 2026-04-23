import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import api from '../../api/axios';
import './Admin.css';

export default function CreateUser() {
    const { t } = useTranslation();
    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [role, setRole] = useState('student');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await api.post('/admin/users', { full_name: fullName, email, password, role });
            navigate('/admin/users');
        } catch (err) {
            setError(err.response?.data?.detail || t('admin.failedCreateUser'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <button className="btn-back" onClick={() => navigate('/admin/users')}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                {t('admin.backToUsers')}
            </button>

            <h1 style={{ marginBottom: '1.5rem' }}>{t('admin.createUserTitle')}</h1>

            <div className="card form-card">
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="fullName">{t('admin.fullName')}</label>
                        <input id="fullName" type="text" value={fullName} onChange={e => setFullName(e.target.value)} placeholder={t('admin.enterFullName')} required />
                    </div>
                    <div className="form-group">
                        <label htmlFor="email">{t('auth.email')}</label>
                        <input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="user@example.com" required />
                    </div>
                    <div className="form-group">
                        <label htmlFor="password">{t('admin.temporaryPassword')}</label>
                        <input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder={t('admin.minChars')} required />
                    </div>
                    <div className="form-group">
                        <label htmlFor="role">{t('admin.role')}</label>
                        <select id="role" value={role} onChange={e => setRole(e.target.value)}>
                            <option value="teacher">{t('admin.teacher')}</option>
                            <option value="student">{t('admin.student')}</option>
                        </select>
                    </div>

                    {error && <div className="alert alert-error">{error}</div>}

                    <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
                        <button type="submit" className="btn-primary" disabled={loading}>
                            {loading ? t('admin.creating') : t('admin.createUserTitle')}
                        </button>
                        <button type="button" className="btn-secondary" onClick={() => navigate('/admin/users')}>
                            {t('teacher.cancel')}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
