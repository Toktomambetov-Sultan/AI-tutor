import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import api from '../../api/axios';
import './Teacher.css';

export default function CourseList() {
    const { t } = useTranslation();
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [showForm, setShowForm] = useState(false);
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');

    const fetchCourses = async () => {
        try {
            const response = await api.get('/courses');
            setCourses(response.data.courses);
        } catch (err) {
            setError(err.response?.data?.detail || t('teacher.failedFetchCourses'));
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchCourses(); }, []);

    const handleCreate = async (e) => {
        e.preventDefault();
        try {
            await api.post('/courses', { title, description });
            setTitle('');
            setDescription('');
            setShowForm(false);
            fetchCourses();
        } catch (err) {
            setError(err.response?.data?.detail || t('teacher.failedCreateCourse'));
        }
    };

    const handleDelete = async (courseId) => {
        if (!confirm(t('teacher.deleteCourseConfirm'))) return;
        try {
            await api.delete(`/courses/${courseId}`);
            fetchCourses();
        } catch (err) {
            setError(err.response?.data?.detail || t('teacher.failedDeleteCourse'));
        }
    };

    return (
        <div className="teacher-course-list-page">
            <div className="teacher-page-header">
                <h1>{t('teacher.myCourses')}</h1>
                <p>{t('teacher.myCoursesDesc')}</p>
            </div>

            <div className="teacher-course-list-layout">
                <aside className="teacher-course-controls">
                    <div className="card teacher-control-card">
                        <h3>{t('teacher.courseControls')}</h3>
                        <p>{t('teacher.courseControlsDesc')}</p>
                        <button className="btn-primary" onClick={() => setShowForm(!showForm)}>
                            {showForm ? t('teacher.cancel') : t('teacher.newCourse')}
                        </button>
                    </div>

                    {showForm && (
                        <div className="inline-form-card teacher-create-form-card">
                            <form onSubmit={handleCreate}>
                                <div className="form-group">
                                    <label>{t('teacher.courseTitle')}</label>
                                    <input type="text" value={title} onChange={e => setTitle(e.target.value)} placeholder="e.g., Introduction to Computer Science" required />
                                </div>
                                <div className="form-group">
                                    <label>{t('teacher.description')}</label>
                                    <textarea value={description} onChange={e => setDescription(e.target.value)} rows={3} placeholder={t('teacher.optionalDesc')} />
                                </div>
                                <button type="submit" className="btn-primary">{t('teacher.createCourse')}</button>
                            </form>
                        </div>
                    )}
                </aside>

                <section className="teacher-course-results">
                    {error && <div className="alert alert-error">{error}<button className="alert-close" onClick={() => setError('')}>×</button></div>}

                    {loading ? (
                        <div className="loading-state"><div className="spinner" /><span>{t('teacher.loadingStudents')}</span></div>
                    ) : courses.length === 0 ? (
                        <div className="empty-state-card">
                            <div className="empty-icon">📚</div>
                            <h3>{t('teacher.noStudents')}</h3>
                            <p>{t('teacher.noStudentsText')}</p>
                        </div>
                    ) : (
                        <div className="course-grid teacher-course-grid">
                            {courses.map(course => (
                                <div key={course.id} className="card course-card">
                                    <h3><Link to={`/teacher/courses/${course.id}`}>{course.title}</Link></h3>
                                    <p className="course-desc">{course.description || t('teacher.failedFetchCourses')}</p>
                                    <div className="course-footer">
                                        <span className="course-date">Created {new Date(course.created_at).toLocaleDateString()}</span>
                                        <button onClick={() => handleDelete(course.id)} className="btn-ghost-danger btn-sm">{t('app.logout')}</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
                </div>
        </div>
    );
}
