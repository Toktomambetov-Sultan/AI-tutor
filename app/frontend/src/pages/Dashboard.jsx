import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

export default function Dashboard() {
    const { user, logout } = useAuthStore();

    if (!user) {
        logout();
        return <Navigate to="/" replace />;
    }

    switch (user.role) {
        case 'admin':
            return <Navigate to="/admin/users" replace />;
        case 'teacher':
            return <Navigate to="/teacher/courses" replace />;
        case 'student':
            return <Navigate to="/courses" replace />;
        default:
            return <Navigate to="/" replace />;
    }
}
