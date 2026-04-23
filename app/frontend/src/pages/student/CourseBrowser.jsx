import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import api from '../../api/axios';
import './Student.css';

export default function CourseBrowser() {
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const { t } = useTranslation();

    useEffect(() => { fetchCourses(); }, []);

    const fetchCourses = async () => {
        try {
            const response = await api.get('/courses');
            setCourses(response.data.courses);
        } catch (err) {
            setError(err.response?.data?.detail || t('student.loadingCourses'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="student-courses-page">
            <div className="student-courses-hero">
                <div>
                    <h1>{t('student.browseCourses')}</h1>
                    <p>{t('student.pickCourse')}</p>
                </div>
                <span className="student-courses-count">{t('student.coursesCount', { count: courses.length })}</span>
            </div>

            {error && <div className="alert alert-error">{error}</div>}

            {loading ? (
                <div className="loading-state"><div className="spinner" /><span>{t('student.loadingCourses')}</span></div>
            ) : courses.length === 0 ? (
                <div className="empty-state-card">
                    <div className="empty-icon">📖</div>
                    <h3>{t('student.noCourses')}</h3>
                    <p>{t('student.noCoursesText')}</p>
                </div>
            ) : (
                <div className="course-grid">
                    {courses.map(course => (
                        <Link to={`/courses/${course.id}`} key={course.id} className="card course-card-link">
                            <h3>{course.title}</h3>
                            <p className="course-desc">{course.description || t('student.noDescription')}</p>
                            <div className="course-meta">
                                <span>👤 {course.teacher_name || t('student.unknownTeacher')}</span>
                                <span>{new Date(course.created_at).toLocaleDateString()}</span>
                            </div>
                        </Link>
                    ))}
                </div>
            )}
        </div>
    );
}
