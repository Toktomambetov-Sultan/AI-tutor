import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

export default function ProtectedRoute({ children, allowedRoles }) {
    const { isAuthenticated, user, logout } = useAuthStore();

    if (!isAuthenticated || !user) {
        // Clear any stale token that lacks a user object
        if (!user && isAuthenticated) logout();
        return <Navigate to="/" replace />;
    }

    if (allowedRoles && !allowedRoles.includes(user.role)) {
        return <Navigate to="/dashboard" replace />;
    }

    return children;
}
