import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, Link, useNavigate } from 'react-router-dom';
import api from '../../api/axios';
import './Student.css';

export default function CourseOverview() {
    const { id } = useParams();
    const { t } = useTranslation();
    const navigate = useNavigate();
    const [course, setCourse] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => { fetchCourse(); }, [id]);

    const fetchCourse = async () => {
        try {
            const response = await api.get(`/courses/${id}`);
            setCourse(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || t('student.courseNotFound'));
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="loading-state"><div className="spinner" /><span>{t('student.loadingCourse')}</span></div>;
    if (!course) return <div className="alert alert-error">{error || t('student.courseNotFound')}</div>;

    const totalLessons = course.classes?.reduce((a, c) => a + (c.lessons?.length || 0), 0) || 0;

    return (
        <div className="course-overview student-course-overview-page">
            <div className="student-overview-topbar">
                <button className="btn-back" onClick={() => navigate('/courses')}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    {t('student.backToCourses')}
                </button>
            </div>

            <div className="student-overview-layout">
                <aside className="student-overview-rail">
                    <div className="course-overview-header">
                        <h1>{course.title}</h1>
                        <div className="course-teacher">
                            👤 {course.teacher_name || t('student.unknownTeacher')}
                        </div>
                        {course.description && (
                            <p className="course-overview-desc">{course.description}</p>
                        )}
                    </div>

                    <div className="card student-overview-stats">
                        <h4>{t('student.snapshot')}</h4>
                        <div className="student-overview-chips">
                            <span className="stat-chip">📁 {t('student.classesCount', { count: course.classes?.length || 0 })}</span>
                            <span className="stat-chip">📖 {t('student.lessonsCount', { count: totalLessons })}</span>
                        </div>
                    </div>
                </aside>

                <section className="course-structure">
                    <h2>{t('student.courseContent')}</h2>

                    {!course.classes?.length ? (
                        <div className="empty-state-card">
                            <div className="empty-icon">📝</div>
                            <h3>{t('student.noContentYet')}</h3>
                            <p>{t('student.noContentYetText')}</p>
                        </div>
                    ) : (
                        course.classes.map(cls => (
                            <div key={cls.id} className="class-block">
                                <div className="class-header">
                                    📁 {cls.title}
                                </div>
                                {!cls.lessons?.length ? (
                                    <p className="no-content-msg">{t('student.noLessonsYet')}</p>
                                ) : (
                                    cls.lessons.map(lesson => (
                                        <div key={lesson.id} className="lesson-item">
                                            <Link to={`/courses/${id}/lessons/${lesson.id}`}>
                                                📖 {lesson.title}
                                            </Link>
                                        </div>
                                    ))
                                )}
                            </div>
                        ))
                    )}
                </section>
            </div>
        </div>
    );
}
