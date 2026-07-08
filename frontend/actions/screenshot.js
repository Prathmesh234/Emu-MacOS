// Action: Screenshot — captures the full screen INCLUDING the Emu overlay.
//
// The model must see the Emu panel so it knows to avoid clicking behind it.
// Uses Electron's desktopCapturer API (runs in-process, inherits the app's
// Screen Recording TCC permission — no child-process permission issues).
//
// The capture is resized to max 1280px width and JPEG-compressed in-process
// so there's no shell piping of base64 data.

const { ipcRenderer } = require('electron');
const path = require('path');
const fs = require('fs');

// desktopCapturer is main-process-only in Electron 12+.
// Imported lazily inside register() which runs in main process.
let desktopCapturer = null;
let nativeImageModule = null;

// Cursor overlay config
const CURSOR_SCALE = 1.0;  // ~regular cursor size, slightly larger for model clarity
const CURSOR_FILL = { r: 255, g: 255, b: 255 };    // white fill
const CURSOR_OUTLINE = { r: 255, g: 20, b: 20 };   // bright red outline

// Arrow cursor shape — polygon points relative to tip (0,0), scaled later
// Classic macOS-style arrow cursor
const CURSOR_ARROW = [
    { x: 0,  y: 0 },
    { x: 0,  y: 20 },
    { x: 5,  y: 16 },
    { x: 9,  y: 24 },
    { x: 12, y: 23 },
    { x: 8,  y: 15 },
    { x: 14, y: 15 },
];

// Screen dimensions for denormalizing [0,1] coordinates → absolute pixels
let _screenWidth = 1920;
let _screenHeight = 1080;

// Origin of the active/locked display in the global virtual-desktop
// coordinate space. Needed so mouse actions land on the same monitor
// Emu is running on, not the primary display.
let _displayOffsetX = 0;
let _displayOffsetY = 0;

// Scale factors for coordinate translation (image coords → screen coords)
let _scaleX = 1;
let _scaleY = 1;

function getScaleFactors() {
    return { scaleX: _scaleX, scaleY: _scaleY };
}

function getScreenDimensions() {
    return { screenWidth: _screenWidth, screenHeight: _screenHeight };
}

function getDisplayOffset() {
    return { displayOffsetX: _displayOffsetX, displayOffsetY: _displayOffsetY };
}

async function captureScreenshot() {
    const result = await ipcRenderer.invoke('screenshot:capture');
    if (result.success && result.imageWidth && result.screenWidth) {
        _scaleX = result.screenWidth / result.imageWidth;
        _scaleY = result.screenHeight / result.imageHeight;
        _screenWidth = result.screenWidth;
        _screenHeight = result.screenHeight;
        if (typeof result.displayOffsetX === 'number') _displayOffsetX = result.displayOffsetX;
        if (typeof result.displayOffsetY === 'number') _displayOffsetY = result.displayOffsetY;
        console.log(`[screenshot] scale ${_scaleX.toFixed(2)}x,${_scaleY.toFixed(2)}y | screen ${_screenWidth}x${_screenHeight} @ offset(${_displayOffsetX},${_displayOffsetY})`);
    }
    return result;
}

/**
 * Test if point (px, py) is inside a polygon using ray-casting algorithm.
 */
function pointInPolygon(px, py, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].x, yi = polygon[i].y;
        const xj = polygon[j].x, yj = polygon[j].y;
        if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
            inside = !inside;
        }
    }
    return inside;
}

/**
 * Minimum distance from point (px, py) to a line segment (ax,ay)-(bx,by).
 */
function distToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return Math.hypot(px - ax, py - ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/**
 * Minimum distance from a point to the polygon border (all edges).
 */
function distToPolygonEdge(px, py, polygon) {
    let minD = Infinity;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const d = distToSegment(px, py, polygon[j].x, polygon[j].y, polygon[i].x, polygon[i].y);
        if (d < minD) minD = d;
    }
    return minD;
}

/**
 * Draw an arrow-shaped cursor onto a raw RGBA bitmap buffer.
 * tipX, tipY = cursor tip position in image pixels.
 */
function drawCursorOverlay(bitmap, imgW, imgH, tipX, tipY) {
    // Scale and translate the arrow polygon to image coordinates
    const poly = CURSOR_ARROW.map(p => ({
        x: tipX + p.x * CURSOR_SCALE,
        y: tipY + p.y * CURSOR_SCALE,
    }));

    // Bounding box for the scan
    const minX = Math.floor(Math.min(...poly.map(p => p.x)) - 3);
    const maxX = Math.ceil(Math.max(...poly.map(p => p.x)) + 3);
    const minY = Math.floor(Math.min(...poly.map(p => p.y)) - 3);
    const maxY = Math.ceil(Math.max(...poly.map(p => p.y)) + 3);

    const OUTLINE_WIDTH = 2;

    for (let py = Math.max(0, minY); py <= Math.min(imgH - 1, maxY); py++) {
        for (let px = Math.max(0, minX); px <= Math.min(imgW - 1, maxX); px++) {
            const offset = (py * imgW + px) * 4;
            if (pointInPolygon(px, py, poly)) {
                // Inside the arrow — white fill
                bitmap[offset]     = CURSOR_FILL.r;
                bitmap[offset + 1] = CURSOR_FILL.g;
                bitmap[offset + 2] = CURSOR_FILL.b;
                bitmap[offset + 3] = 255;
            } else if (distToPolygonEdge(px, py, poly) <= OUTLINE_WIDTH) {
                // Near the edge — red outline
                bitmap[offset]     = CURSOR_OUTLINE.r;
                bitmap[offset + 1] = CURSOR_OUTLINE.g;
                bitmap[offset + 2] = CURSOR_OUTLINE.b;
                bitmap[offset + 3] = 255;
            }
        }
    }
}

/**
 * Overlay the current cursor onto a nativeImage screenshot.
 *
 * Resolution-agnostic: uses Electron's `screen.getCursorScreenPoint()` and
 * `activeDisplay.bounds/size`, both of which live in the same DIP
 * coordinate space. That lets the math automatically adapt to whatever
 * display Emu is running on — 1080p, 1440p, Retina, an external 4K, or a
 * mixed-DPI multi-monitor rig — without hard-coding any resolution.
 *
 * Why not `cliclick p:.`? On multi-monitor macOS setups where displays
 * have different scale factors, cliclick's CoreGraphics coordinates don't
 * always agree with Electron's DIP space, which caused the cursor to fall
 * outside the captured image and be silently dropped.
 *
 * `displayOffsetX/Y` are the captured display's origin (DIPs), so the
 * cursor is first translated into display-local coords, then proportionally
 * mapped to the image's pixel grid via (imgW/screenW, imgH/screenH).
 */
async function overlayMouseCursor(image, screen, screenW, screenH, displayOffsetX = 0, displayOffsetY = 0) {
    try {
        const { x: cursorX, y: cursorY } = screen.getCursorScreenPoint();

        const imgSize = image.getSize();
        const relX = cursorX - displayOffsetX;
        const relY = cursorY - displayOffsetY;

        // Proportional map: works for any screen resolution / image size.
        const imgX = Math.round((relX / screenW) * imgSize.width);
        const imgY = Math.round((relY / screenH) * imgSize.height);

        const onDisplay = relX >= 0 && relY >= 0 && relX < screenW && relY < screenH;
        console.log(`[screenshot] cursor screen(${cursorX},${cursorY}) rel(${relX},${relY}) of ${screenW}×${screenH} → image(${imgX},${imgY}) of ${imgSize.width}×${imgSize.height}  onCapturedDisplay=${onDisplay}`);

        if (!onDisplay) {
            // Cursor is on a different monitor than the one being captured —
            // nothing to draw, and drawing would land out of bounds.
            console.warn('[screenshot] cursor is OFF the captured display — overlay skipped');
            return image;
        }

        // Get raw RGBA bitmap, draw cursor, create new image
        const bitmap = image.toBitmap();
        drawCursorOverlay(bitmap, imgSize.width, imgSize.height, imgX, imgY);

        const result = nativeImageModule.createFromBitmap(bitmap, {
            width: imgSize.width,
            height: imgSize.height,
        });
        return result;
    } catch (err) {
        console.warn(`[screenshot] cursor overlay failed, skipping: ${err.message}`);
        return image;
    }
}

function register(ipcMain, { screen, getMainWindow }) {
    // Import desktopCapturer here — this runs in the main process
    desktopCapturer = require('electron').desktopCapturer;
    nativeImageModule = require('electron').nativeImage;
    console.log(`[screenshot] register: desktopCapturer available = ${!!desktopCapturer}`);
    if (process.platform === 'darwin') {
        try {
            const status = require('electron').systemPreferences.getMediaAccessStatus('screen');
            console.log(`[screenshot] Screen Recording permission status: ${status}`);
        } catch (err) {
            console.warn(`[screenshot] could not query Screen Recording status: ${err.message}`);
        }
    }

    // Live active-display info. Unlike getScreenDimensions() / getDisplayOffset()
    // — which cache the most recent screenshot's values — this always reflects
    // the currently locked (or computed) display, so mouse actions remain
    // correct even when dispatched before the first screenshot of a session
    // or after the user moves Emu to a different monitor mid-session.
    ipcMain.handle('display:info', async () => {
        try {
            const { getActiveDisplay, getLockedDisplay } = require('../display/activeDisplay');
            const d = getLockedDisplay() || getActiveDisplay(screen, getMainWindow());
            return {
                success: true,
                screenWidth:  d.size.width,
                screenHeight: d.size.height,
                displayOffsetX: d.bounds.x,
                displayOffsetY: d.bounds.y,
                scaleFactor:  d.scaleFactor || 1,
                displayId:    d.id,
            };
        } catch (err) {
            return { success: false, error: err.message };
        }
    });

    ipcMain.handle('screenshot:capture', async () => {
        try {
            if (!desktopCapturer) {
                throw new Error('desktopCapturer not available — Electron main process API missing');
            }

            // macOS TCC preflight: without Screen Recording permission,
            // getSources() returns no/black sources with no explanation.
            // Fail fast with an actionable message instead.
            if (process.platform === 'darwin') {
                const status = require('electron').systemPreferences.getMediaAccessStatus('screen');
                if (status !== 'granted') {
                    throw new Error(
                        `Screen Recording permission is "${status}" for this app. ` +
                        'Grant it in System Settings → Privacy & Security → Screen Recording ' +
                        '(add/enable the Emu/Electron app), then restart Emu.'
                    );
                }
            }

            const { getActiveDisplay, getLockedDisplay } = require('../display/activeDisplay');
            const win = getMainWindow();
            const activeDisplay = getLockedDisplay() || getActiveDisplay(screen, win);
            const { width: screenWidth, height: screenHeight } = activeDisplay.size;
            const scaleFactor = activeDisplay.scaleFactor || 1;
            // The display's origin in the virtual-desktop DIP coordinate space.
            // Both capture paths below produce a display-local image (covering
            // exactly this display from (0,0) to (screenWidth, screenHeight)),
            // so the cursor-overlay math consistently uses this offset to
            // translate virtual-desktop cursor coords → image-local coords.
            // Mouse actions also use this offset so clicks land on the right
            // monitor regardless of which display Emu is running on.
            const { x: displayOffsetX, y: displayOffsetY } = activeDisplay.bounds;

            console.log(`[screenshot] capturing display ${activeDisplay.id}: ${screenWidth}×${screenHeight}, scale ${scaleFactor}x, offset (${displayOffsetX},${displayOffsetY})`);

            // Capture via desktopCapturer — grabs the full screen including all windows
            const sources = await desktopCapturer.getSources({
                types: ['screen'],
                thumbnailSize: {
                    width: Math.round(screenWidth * scaleFactor),
                    height: Math.round(screenHeight * scaleFactor),
                },
            });

            if (!sources || sources.length === 0) {
                throw new Error('desktopCapturer returned no sources — check Screen Recording permission');
            }

            // Match source to active display by display_id; fall back to first source
            const targetId = String(activeDisplay.id);
            const matched = sources.find(s => String(s.display_id) === targetId);
            if (!matched) {
                console.warn(`[screenshot] display_id ${targetId} not matched in sources (ids: ${sources.map(s => s.display_id).join(', ')}); falling back to sources[0]`);
            }
            const source = matched || sources[0];
            const nativeImage = source.thumbnail;

            let finalImage = nativeImage;

            if (nativeImage.isEmpty()) {
                console.warn('[screenshot] desktopCapturer returned empty image (possibly due to Hardware Acceleration/DRM). Falling back to native macOS screencapture...');
                const os = require('os');
                const cp = require('child_process');
                const tmpPath = path.join(os.tmpdir(), `emu_fallback_${Date.now()}.png`);

                try {
                    // -D <n> targets a specific display (1-based index in getAllDisplays order).
                    // This ensures the fallback captures the same display as the primary path.
                    const allDisplays = screen.getAllDisplays();
                    const displayIndex = allDisplays.findIndex(d => d.id === activeDisplay.id);
                    const displayFlag = displayIndex >= 0 ? `-D ${displayIndex + 1}` : '';
                    if (!displayFlag) {
                        console.warn('[screenshot] could not determine display index for screencapture fallback; capturing default display');
                    }
                    cp.execSync(`screencapture -x ${displayFlag} "${tmpPath}"`);
                    finalImage = nativeImageModule.createFromPath(tmpPath);
                    if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);

                    if (finalImage.isEmpty()) {
                        throw new Error('Fallback native screencapture also returned an empty image');
                    }
                    // NOTE: The -D fallback image is already display-local
                    // (covers exactly this display from (0,0) to its size).
                    // displayOffsetX/Y must REMAIN the display's global
                    // origin so `cursor - offset` correctly produces
                    // image-local coords for the cursor overlay.
                } catch (fallbackErr) {
                    throw new Error(`desktopCapturer failed (empty image) and screencapture fallback failed: ${fallbackErr.message}`);
                }
            }

            // Resize to max 1280px width to keep under Claude's size limit
            const size = finalImage.getSize();
            if (size.width > 1280) {
                const newHeight = Math.round((1280 / size.width) * size.height);
                finalImage = finalImage.resize({ width: 1280, height: newHeight, quality: 'good' });
            }

            // Overlay a cursor indicator so the model can see pointer position
            finalImage = await overlayMouseCursor(finalImage, screen, screenWidth, screenHeight, displayOffsetX, displayOffsetY);

            const finalSize = finalImage.getSize();

            // JPEG at 80% quality — good balance of size and clarity
            const jpegBuffer = finalImage.toJPEG(80);
            const base64 = jpegBuffer.toString('base64');

            console.log(`[screenshot] captured ${finalSize.width}×${finalSize.height} (screen: ${screenWidth}×${screenHeight}, scale: ${scaleFactor}x) → ${Math.round(base64.length / 1024)} KB`);

            return {
                success: true,
                screenWidth,
                screenHeight,
                imageWidth: finalSize.width,
                imageHeight: finalSize.height,
                displayOffsetX,
                displayOffsetY,
                base64,
            };
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            console.error('[screenshot] capture failed:', msg, err);
            return { success: false, error: msg };
        }
    });
}

module.exports = { captureScreenshot, register, getScaleFactors, getScreenDimensions, getDisplayOffset };
