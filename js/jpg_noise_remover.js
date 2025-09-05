import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SnJake.SnJakeArtifactsRemover",
    async nodeCreated(node) {
        if (node.comfyClass === "SnJakeArtifactsRemover") {
            node.color = "#2e2e36";
            node.bgcolor = "#41414a";
        }
    }
});