package dev.notune.transcribe;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Which speech models the app can use: the bundled Parakeet model (absent in
 * F-Droid builds) and GGUF files imported into {@code files/models/}. The
 * selection is the {@code active_model} marker file read by the Rust engine
 * loader: the file name of an imported model, absent/empty = bundled model.
 */
public final class ModelStore {
    private static final String TAG = "ModelStore";
    private static final String ACTIVE_MODEL_FILE = "active_model";
    /** Asset directory of the bundled model (see src/assets.rs). */
    private static final String BUILTIN_ASSET_DIR = "builtin-model";

    /** Engine status when there is nothing to load (NO_MODEL_STATUS in src/engine.rs). */
    public static final String NO_MODEL_STATUS = "Error: no speech model installed";

    /** The bundled model, offered as the first download when it isn't bundled. */
    public static final String RECOMMENDED_NAME = "Parakeet TDT 0.6B v3";
    public static final String RECOMMENDED_SIZE = "485 MB";
    public static final String RECOMMENDED_URL =
            "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q4_K_M.gguf";

    private static Boolean builtin;

    private ModelStore() {}

    /** True if this build bundles a model (F-Droid builds don't). */
    public static synchronized boolean hasBuiltinModel(Context ctx) {
        if (builtin == null) {
            boolean found = false;
            try {
                String[] names = ctx.getAssets().list(BUILTIN_ASSET_DIR);
                if (names != null) {
                    for (String name : names) {
                        if (isGguf(name)) found = true;
                    }
                }
            } catch (IOException e) {
                Log.w(TAG, "Failed to list bundled model assets", e);
            }
            builtin = found;
        }
        return builtin;
    }

    public static boolean hasAnyModel(Context ctx) {
        return hasBuiltinModel(ctx) || !importedModels(ctx).isEmpty()
                || hasLanguageModels(ctx);
    }

    /**
     * Directory for the per-language models ({@code Android/media/<package>/models/}),
     * which are copied onto the device next to the APK instead of being
     * bundled or imported. The native engine picks the file for the language
     * setting; see {@code LANGUAGE_MODELS} in {@code src/engine.rs}.
     */
    public static File languageModelsDir(Context ctx) {
        File[] media = ctx.getExternalMediaDirs();
        if (media.length == 0 || media[0] == null) return null;
        return new File(media[0], "models");
    }

    /** True if any per-language model file is present. */
    public static boolean hasLanguageModels(Context ctx) {
        return !languageModels(ctx).isEmpty();
    }

    public static File modelsDir(Context ctx) {
        File dir = new File(ctx.getFilesDir(), "models");
        dir.mkdirs();
        return dir;
    }

    /** File names of the imported models, sorted for display. */
    public static List<String> importedModels(Context ctx) {
        List<String> names = new ArrayList<>();
        File[] files = modelsDir(ctx).listFiles((dir, name) -> isModelFile(name));
        if (files != null) {
            for (File f : files) names.add(f.getName());
        }
        names.sort(String.CASE_INSENSITIVE_ORDER);
        return names;
    }

    static boolean isGguf(String name) {
        return name.toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    /**
     * True for files the engine can load: GGUF, or a legacy whisper.cpp
     * {@code .bin}, which transcribe.cpp reads directly.
     */
    static boolean isModelFile(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".gguf") || lower.endsWith(".bin");
    }

    /** File names in the per-language models directory, sorted for display. */
    public static List<String> languageModels(Context ctx) {
        List<String> names = new ArrayList<>();
        File dir = languageModelsDir(ctx);
        File[] files = dir == null ? null : dir.listFiles((d, name) -> isModelFile(name));
        if (files != null) {
            for (File f : files) names.add(f.getName());
        }
        names.sort(String.CASE_INSENSITIVE_ORDER);
        return names;
    }

    /** The selected imported model, or "" for the bundled one. */
    public static String activeModel(Context ctx) {
        File f = new File(ctx.getFilesDir(), ACTIVE_MODEL_FILE);
        if (!f.exists()) return "";
        try {
            return new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8).trim();
        } catch (IOException e) {
            return "";
        }
    }

    /** Selects an imported model; null or "" selects the bundled one. */
    public static void setActiveModel(Context ctx, String fileName) {
        File f = new File(ctx.getFilesDir(), ACTIVE_MODEL_FILE);
        if (fileName == null || fileName.isEmpty()) {
            f.delete();
            return;
        }
        try {
            Files.write(f.toPath(), fileName.getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            Log.e(TAG, "Failed to write " + ACTIVE_MODEL_FILE, e);
        }
    }

    /**
     * Keeps the selection pointing at a model that exists: a deleted model
     * falls back to the bundled one, or without one to the first imported
     * model. Returns the (possibly updated) selection.
     */
    public static String ensureValidSelection(Context ctx) {
        String active = activeModel(ctx);
        List<String> names = importedModels(ctx);
        if (!active.isEmpty() && names.contains(active)) return active;
        if (active.isEmpty() && hasBuiltinModel(ctx)) return active;

        String fallback = hasBuiltinModel(ctx) || names.isEmpty() ? "" : names.get(0);
        if (!fallback.equals(active)) setActiveModel(ctx, fallback);
        return fallback;
    }

    /** An engine status for display, with the no-model error translated. */
    public static String displayStatus(Context ctx, String status) {
        return NO_MODEL_STATUS.equals(status) ? ctx.getString(R.string.no_model_short) : status;
    }
}
