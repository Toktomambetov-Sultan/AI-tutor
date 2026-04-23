import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/axios';
import './Student.css';

export default function MaterialViewer() {
    const { id: courseId, lessonId, materialId } = useParams();
    const { t } = useTranslation();
    const navigate = useNavigate();
    const [material, setMaterial] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchMaterial = async () => {
            try {
                const response = await api.get(`/materials/${materialId}`);
                setMaterial(response.data);
            } catch (err) {
                setError(err.response?.data?.detail || t('student.materialNotFound'));
            } finally {
                setLoading(false);
            }
        };

        fetchMaterial();
    }, [materialId]);

    if (loading) {
        return (
            <div className="lesson-viewer lesson-viewer-layout">
                <div className="loading-state"><div className="spinner" /><span>{t('student.loadingMaterial')}</span></div>
            </div>
        );
    }

    if (!material) {
        return (
            <div className="lesson-viewer lesson-viewer-layout">
                <div className="alert alert-error">{error || t('student.materialNotFound')}</div>
                <button className="btn-back" onClick={() => navigate(`/courses/${courseId}/lessons/${lessonId}`)}>
                    {t('student.backToLesson')}
                </button>
            </div>
        );
    }

    return (
        <div className="lesson-viewer lesson-viewer-layout material-viewer-page">
            <div className="lesson-topbar">
                <button className="btn-back" onClick={() => navigate(`/courses/${courseId}/lessons/${lessonId}`)}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    {t('student.backToLesson')}
                </button>
                <h1>{material.type === 'text' ? t('student.materialTitle') : t('student.viewMaterial')}</h1>
            </div>

            <div className="card material-block">
                {material.type === 'text' ? (
                    <div className="text-content">{material.content}</div>
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
        </div>
    );
}
