import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../store/authStore';
import LanguageSwitcher from './LanguageSwitcher';
import './Layout.css';

export default function Layout({ children }) {
    const { user, logout } = useAuthStore();
    const { t } = useTranslation();
    const navigate = useNavigate();
    const location = useLocation();
    const [mobileOpen, setMobileOpen] = useState(false);

    const handleLogout = async () => {
        await logout();
        navigate('/');
    };

    const getNavLinks = () => {
        if (!user) return [];
        switch (user.role) {
            case 'admin':
                return [{ to: '/admin/users', label: t('nav.users'), icon: '👥' }];
            case 'teacher':
                return [
                    { to: '/teacher/courses', label: t('nav.myCourses'), icon: '📚' },
                    { to: '/teacher/students', label: t('nav.students'), icon: '🎓' },
                ];
            case 'student':
                return [{ to: '/courses', label: t('nav.courses'), icon: '📖' }];
            default:
                return [];
        }
    };

    const links = getNavLinks();
    const initials = user?.full_name?.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) || '?';

    return (
        <div className="layout">
            <header className="header">
                <div className="container header-content">
                    <Link to="/dashboard" className="logo">
                        <span className="logo-icon">🎓</span>
                        {t('app.name')}
                    </Link>

                    <nav className="nav">
                        {links.map(link => (
                            <Link
                                key={link.to}
                                to={link.to}
                                className={`nav-link ${location.pathname.startsWith(link.to) ? 'active' : ''}`}
                            >
                                {link.icon} {link.label}
                            </Link>
                        ))}
                    </nav>

                    <div className="header-right">
                        <LanguageSwitcher />
                        <div className="user-chip">
                            <div className="user-avatar">{initials}</div>
                            <span className="user-name">{user?.full_name}</span>
                            <span className={`badge badge-${user?.role}`}>{t(`roles.${user?.role}`)}</span>
                        </div>
                        <button onClick={handleLogout} className="logout-btn">{t('app.logout')}</button>
                    </div>

                    <button className="mobile-toggle" onClick={() => setMobileOpen(!mobileOpen)}>
                        {mobileOpen ? (
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
                        ) : (
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 7h18M3 12h18M3 17h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
                        )}
                    </button>
                </div>

                {mobileOpen && (
                    <nav className="mobile-nav">
                        {links.map(link => (
                            <Link
                                key={link.to}
                                to={link.to}
                                className={`nav-link ${location.pathname.startsWith(link.to) ? 'active' : ''}`}
                                onClick={() => setMobileOpen(false)}
                            >
                                {link.icon} {link.label}
                            </Link>
                        ))}
                    </nav>
                )}
            </header>
            <main className="main container">
                {children}
            </main>
        </div>
    );
}
