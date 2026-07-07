// Action: Left Click
// Uses cliclick on macOS, xdotool on Linux.
const { ipcRenderer } = require('electron');
const psProcess = require('../process/psProcess');

const isMac = process.platform === 'darwin';

async function leftClick(x, y) {
    return await ipcRenderer.invoke('mouse:left-click', { x, y });
}

function register(ipcMain, BACKEND_URL) {
    ipcMain.handle('mouse:left-click', async (_event, { x, y }) => {
        try {
            const cmd = isMac
                ? `cliclick c:.`   // single-click at current cursor position
                : `xdotool click 1`;
            await psProcess.run(cmd);
            return { success: true, x, y };
        } catch (err) {
            return { success: false, error: err.message };
        }
    });
}

module.exports = { leftClick, register };
