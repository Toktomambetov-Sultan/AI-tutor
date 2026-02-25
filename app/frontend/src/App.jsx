import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import ProtectedRoute from './components/ProtectedRoute';
import Layout from './components/Layout';

// Public pages
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';

// Admin pages
import UserManagement from './pages/admin/UserManagement';
import CreateUser from './pages/admin/CreateUser';
import UserDetail from './pages/admin/UserDetail';

// Teacher pages
import CourseList from './pages/teacher/CourseList';
import CourseDetail from './pages/teacher/CourseDetail';
import StudentList from './pages/teacher/StudentList';

// Student pages
import CourseBrowser from './pages/student/CourseBrowser';
import CourseOverview from './pages/student/CourseOverview';
import LessonViewer from './pages/student/LessonViewer';
import AICall from './pages/student/AICall';

export default function App() {
    const { isAuthenticated } = useAuthStore();

    return (
        <Routes>
            {/* Public routes */}
            <Route path="/" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login />} />
            <Route path="/register" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Register />} />

            {/* Dashboard redirect */}
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />

            {/* Admin routes */}
            <Route path="/admin/users" element={<ProtectedRoute allowedRoles={['admin']}><Layout><UserManagement /></Layout></ProtectedRoute>} />
            <Route path="/admin/users/new" element={<ProtectedRoute allowedRoles={['admin']}><Layout><CreateUser /></Layout></ProtectedRoute>} />
            <Route path="/admin/users/:id" element={<ProtectedRoute allowedRoles={['admin']}><Layout><UserDetail /></Layout></ProtectedRoute>} />

            {/* Teacher routes */}
            <Route path="/teacher/courses" element={<ProtectedRoute allowedRoles={['teacher']}><Layout><CourseList /></Layout></ProtectedRoute>} />
            <Route path="/teacher/courses/:id" element={<ProtectedRoute allowedRoles={['teacher']}><Layout><CourseDetail /></Layout></ProtectedRoute>} />
            <Route path="/teacher/students" element={<ProtectedRoute allowedRoles={['teacher']}><Layout><StudentList /></Layout></ProtectedRoute>} />

            {/* Student routes */}
            <Route path="/courses" element={<ProtectedRoute allowedRoles={['student']}><Layout><CourseBrowser /></Layout></ProtectedRoute>} />
            <Route path="/courses/:id" element={<ProtectedRoute allowedRoles={['student']}><Layout><CourseOverview /></Layout></ProtectedRoute>} />
            <Route path="/courses/:id/lessons/:lessonId" element={<ProtectedRoute allowedRoles={['student']}><Layout><LessonViewer /></Layout></ProtectedRoute>} />
            <Route path="/courses/:id/lessons/:lessonId/call" element={<ProtectedRoute allowedRoles={['student']}><AICall /></ProtectedRoute>} />

            {/* Catch-all */}
            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    );
}
