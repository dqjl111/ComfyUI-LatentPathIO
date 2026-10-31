import { app } from "../../scripts/app.js";

const NO_FILES = "(none found)";

// "Load Latent (Folder)":
//   - latent_name shows FILE NAMES only (no folder prefix)
//   - `folder` scopes the list to that directory, so an audio latent simply
//     cannot be picked on the video node (and vice versa)
//   - values always come from the server's full name list, so ComfyUI's
//     "value not in list" validation always passes
app.registerExtension({
  name: "ComfyUI.LatentPathIO",
  async nodeCreated(node) {
    if (node.comfyClass !== "Load Latent (Folder)") return;

    const folderWidget = node.widgets.find((w) => w.name === "folder");
    const fileWidget = node.widgets.find((w) => w.name === "latent_name");
    if (!folderWidget || !fileWidget) return;

    let items = []; // [{ dir, name }]
    let allNames = Array.isArray(fileWidget.options?.values)
      ? [...fileWidget.options.values]
      : [NO_FILES];
    let outputDir = "";

    const normNeedle = (raw) => {
      let f = (raw || "").trim().replace(/\\/g, "/").toLowerCase();
      if (!f) return "";
      const out = (outputDir || "").replace(/\\/g, "/").toLowerCase().replace(/\/+$/, "");
      if (out && f.startsWith(out)) {
        f = f.slice(out.length);
      } else {
        const i = f.lastIndexOf("/output/");
        if (i >= 0) f = f.slice(i + "/output/".length);
      }
      return f
        .replace(/^[a-z]:\//, "")
        .replace(/^\/+/, "")
        .replace(/^output\//, "")
        .replace(/\/+$/, "");
    };

    const apply = () => {
      const needle = normNeedle(folderWidget.value);
      let names = [];
      if (needle) {
        names = items
          .filter((it) => (it.dir || "").toLowerCase().includes(needle))
          .map((it) => it.name);
      }
      if (!names.length) names = allNames; // folder empty or no match

      names = [...new Set(names)];
      const values = names.length ? names : [NO_FILES];

      fileWidget.options.values = values;
      if (!values.includes(fileWidget.value)) fileWidget.value = values[0];
      node.setDirtyCanvas?.(true, true);
      if (typeof fileWidget.callback === "function") fileWidget.callback(fileWidget.value);
    };

    const refresh = async () => {
      try {
        const resp = await fetch("/latentpathio/list", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        const data = await resp.json();
        if (data) {
          if (Array.isArray(data.items)) items = data.items;
          if (Array.isArray(data.files) && data.files.length) allNames = data.files;
          if (data.output_dir) outputDir = data.output_dir;
        }
      } catch (e) {
        console.warn("[LatentPathIO] 刷新列表失败:", e);
      }
      apply();
    };

    node.addWidget("button", "🔄 刷新 latent 列表", null, () => {
      refresh();
    });

    folderWidget.callback = () => apply();
    apply();
    refresh();
  },
});
