import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../../api/axios';
import './Student.css';

export default function CourseBrowser() {
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => { fetchCourses(); }, []);

    const fetchCourses = async () => {
        try {
            const response = await api.get('/courses');
            setCourses(response.data.courses);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch courses');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <div className="page-header">
                <h1>Browse Courses</h1>
            </div>

            {error && <div className="alert alert-error">{error}</div>}

            {loading ? (
                <div className="loading-state"><div className="spinner" /><span>Loading courses...</span></div>
            ) : courses.length === 0 ? (
                <div className="empty-state-card">
                    <div className="empty-icon">📖</div>
                    <h3>No courses available</h3>
                    <p>Check back later — teachers are preparing great content for you!</p>
                </div>
            ) : (
                <div className="course-grid">
                    {courses.map(course => (
                        <Link to={`/courses/${course.id}`} key={course.id} className="card course-card-link">
                            <h3>{course.title}</h3>
                            <p className="course-desc">{course.description || 'No description'}</p>
                            <div className="course-meta">
                                <span>👤 {course.teacher_name || 'Unknown'}</span>
                                <span>{new Date(course.created_at).toLocaleDateString()}</span>
                            </div>
                        </Link>
                    ))}
                </div>
            )}
        </div>
    );
}
