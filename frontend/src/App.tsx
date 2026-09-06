import { useCallback, useEffect, useState } from "react";
import { Disclaimer } from "./components/Common";
import ClipDetail from "./screens/ClipDetail";
import Processing from "./screens/Processing";
import Results from "./screens/Results";
import Upload from "./screens/Upload";

type View =
  | { name: "upload" }
  | { name: "processing"; projectId: string }
  | { name: "results"; projectId: string }
  | { name: "clip"; projectId: string; clipId: string };

/**
 * Routing is done with the URL hash rather than a router library: there are four
 * screens, and this keeps refresh and back-button behaviour working without a
 * dependency.
 *
 *   #/               upload
 *   #/p/<id>         processing
 *   #/p/<id>/clips   results
 *   #/p/<id>/c/<cid> clip detail
 */
function parseHash(): View {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const parts = hash.split("/").filter(Boolean);

  if (parts[0] === "p" && parts[1]) {
    if (parts[2] === "clips") return { name: "results", projectId: parts[1] };
    if (parts[2] === "c" && parts[3])
      return { name: "clip", projectId: parts[1], clipId: parts[3] };
    return { name: "processing", projectId: parts[1] };
  }
  return { name: "upload" };
}

function navigate(path: string) {
  window.location.hash = path;
}

export default function App() {
  const [view, setView] = useState<View>(parseHash);

  useEffect(() => {
    const onHashChange = () => setView(parseHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const goResults = useCallback((projectId: string) => {
    navigate(`/p/${projectId}/clips`);
  }, []);

  const screen = () => {
    switch (view.name) {
      case "processing":
        return (
          <Processing
            projectId={view.projectId}
            onComplete={() => goResults(view.projectId)}
            onCancel={() => navigate("/")}
          />
        );

      case "results":
        return (
          <Results
            projectId={view.projectId}
            onOpenClip={(clipId) => navigate(`/p/${view.projectId}/c/${clipId}`)}
            onNewProject={() => navigate("/")}
          />
        );

      case "clip":
        return (
          <ClipDetail
            projectId={view.projectId}
            clipId={view.clipId}
            onBack={() => goResults(view.projectId)}
          />
        );

      default:
        return <Upload onStarted={(projectId) => navigate(`/p/${projectId}`)} />;
    }
  };

  // One notice for the whole app rather than one per screen: it applies to
  // everything the app produces, and a reader should not be able to miss it by
  // taking a different route through the UI.
  return (
    <>
      {screen()}
      <Disclaimer />
    </>
  );
}
