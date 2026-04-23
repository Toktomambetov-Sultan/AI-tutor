import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../../api/axios';
import './Student.css';

export default function LessonViewer() {
    const { id: courseId, lessonId } = useParams();
    const { t } = useTranslation();
    const navigate = useNavigate();
    const [lesson, setLesson] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => { fetchLesson(); }, [lessonId]);

    const fetchLesson = async () => {
        try {
            const response = await api.get(`/lessons/${lessonId}`);
            setLesson(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || t('student.lessonNotFound'));
        } finally {
            setLoading(false);
        }
    };

    const handleAiCall = () => {
        navigate(`/courses/${courseId}/lessons/${lessonId}/call`);
    };

    if (loading) return <div className="loading-state"><div className="spinner" /><span>{t('student.loadingLesson')}</span></div>;
    if (!lesson) return <div className="alert alert-error">{error || t('student.lessonNotFound')}</div>;

    return (
        <div className="lesson-viewer lesson-viewer-layout">
            <div className="lesson-topbar">
                <button className="btn-back" onClick={() => navigate(`/courses/${courseId}`)}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    {t('student.backToCourse')}
                </button>
                <h1>{lesson.title}</h1>
            </div>

            <div className="lesson-columns">
                <section className="lesson-main-column">
                    {!lesson.materials?.length ? (
                        <div className="empty-state-card" style={{ marginTop: '1rem' }}>
                            <div className="empty-icon">📝</div>
                            <h3>{t('student.noMaterialsYet')}</h3>
                            <p>{t('student.noMaterialsYetText')}</p>
                        </div>
                    ) : (
                        lesson.materials.map((material, index) => (
                            <div key={material.id} className="card material-block">
                                <h3>
                                    {material.type === 'text' ? t('student.textMaterial') : t('student.pdfDocument')}
                                    {lesson.materials.length > 1 && ` #${index + 1}`}
                                </h3>
                                {material.type === 'text' ? (
                                    <button
                                        className="btn-secondary"
                                        onClick={() => navigate(`/courses/${courseId}/lessons/${lessonId}/material/${material.id}`)}
                                    >
                                        {t('student.viewMaterial')}
                                    </button>
                                ) : (
                                    <a
                                        href={`/api/v1/materials/${material.id}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="pdf-link"
                                    >
                                        {t('student.openPdf')}
                                    </a>
                                )}
                            </div>
                        ))
                    )}
                </section>

                <aside className="lesson-side-column">
                    <div className="ai-section">
                        <div className="ai-section-icon">🤖</div>
                        <h3>{t('student.aiTutor')}</h3>
                        <p>{t('student.aiTutorText')}</p>
                        <button className="btn-primary" onClick={handleAiCall}>
                            {t('student.startVoiceSession')}
                        </button>
                    </div>

                    <div className="card lesson-stats-card">
                        <h4>{t('student.lessonSnapshot')}</h4>
                        <p>{t('student.materialsReady', { count: lesson.materials?.length || 0 })}</p>
                    </div>
                </aside>
            </div>
        </div>
    );
}
