import { create } from 'zustand';
import api from '../api/axios';

export const useAuthStore = create((set, get) => ({
    token: localStorage.getItem('token') || null,
    user: JSON.parse(localStorage.getItem('user') || 'null'),
    isAuthenticated: !!localStorage.getItem('token') && !!JSON.parse(localStorage.getItem('user') || 'null'),

    login: async (email, password) => {
        const response = await api.post('/auth/login', { email, password });
        const { token, user_id, role, full_name } = response.data;

        const user = { id: user_id, role, full_name };

        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));

        set({ token, user, isAuthenticated: true });

        return user;
    },

    register: async (full_name, email, password) => {
        await api.post('/auth/register', { full_name, email, password });
    },

    logout: async () => {
        try {
            await api.post('/auth/logout');
        } catch {
            // Token might already be expired
        }
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        set({ token: null, user: null, isAuthenticated: false });
    },

    changePassword: async (current_password, new_password) => {
        await api.patch('/auth/change-password', { current_password, new_password });
    },

    changeUsername: async (new_full_name) => {
        await api.patch('/auth/change-username', { new_full_name });
        const user = { ...get().user, full_name: new_full_name };
        localStorage.setItem('user', JSON.stringify(user));
        set({ user });
    },
}));
