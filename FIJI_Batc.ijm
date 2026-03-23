// ================== 你只需要修改这两行路径 ==================
baseDir   = "//Hive3014/znn/YuanqiHua/High speed/260312 dmrt3 MTZ/";
outputDir = "//Hive3014/znn/YuanqiHua/High speed/260312 dmrt3 MTZ/AVI_out/";
// ============================================================

// ---- 路径标准化：全部用 / ，并保证最后有 / ----
function normPath(p) {
    p = replace(p, "\\", "/");
    while (indexOf(p, "//") == 0 && indexOf(substring(p, 2), "//") >= 0) {
        head = "//";
        tail = substring(p, 2);
        while (indexOf(tail, "//") >= 0)
            tail = replace(tail, "//", "/");
        p = head + tail;
    }
    while (indexOf(p, "//") > 0)
        p = replace(p, "//", "/");
    if (!endsWith(p, "/"))
        p = p + "/";
    return p;
}

baseDir = normPath(baseDir);
outputDir = normPath(outputDir);

File.makeDirectory(outputDir);

function folderHasJpg(dirPath) {
    dirPath = normPath(dirPath);
    lst = getFileList(dirPath);
    for (kk = 0; kk < lst.length; kk++) {
        name = toLowerCase(lst[kk]);
        if (endsWith(name, ".jpg") || endsWith(name, ".jpeg"))
            return true;
    }
    return false;
}

function openStackFromFolderOrSubfolders(folderPath, folderName) {
    folderPath = normPath(folderPath);

    // 情况1：当前文件夹里直接就有 jpg
    if (folderHasJpg(folderPath)) {
        File.openSequence(folderPath);
        return getImageID();
    }

    // 情况2：当前文件夹里没有 jpg，则尝试读取一层子文件夹里的 jpg
    subList = getFileList(folderPath);
    mergedOpened = false;
    mergedTitle = "__MERGED_TMP__";
    mergedID = -1;

    for (ss = 0; ss < subList.length; ss++) {
        subName = subList[ss];

        if (endsWith(subName, "/") || endsWith(subName, "\\"))
            subName2 = substring(subName, 0, lengthOf(subName) - 1);
        else
            subName2 = subName;

        subPath = normPath(folderPath + subName2);

        if (!File.isDirectory(subPath))
            continue;

        if (!folderHasJpg(subPath))
            continue;

        File.openSequence(subPath);

        curID = getImageID();
        curTitle = getTitle();

        if (curTitle == "")
            continue;

        if (!mergedOpened) {
            selectImage(curID);
            rename(mergedTitle);
            mergedID = getImageID();
            mergedOpened = true;
        } else {
            run("Concatenate...", "title=[" + mergedTitle + "] image1=[" + mergedTitle + "] image2=[" + curTitle + "]");

            selectImage(mergedTitle);
            mergedID = getImageID();

            found = false;
            titles = getList("image.titles");
            for (tt = 0; tt < titles.length; tt++) {
                if (titles[tt] == curTitle) {
                    found = true;
                    break;
                }
            }
            if (found) {
                selectImage(curTitle);
                close();
            }

            selectImage(mergedID);
        }
    }

    if (mergedOpened) {
        selectImage(mergedID);
        rename(folderName);
        return mergedID;
    }

    return -1;
}

setBatchMode(true);

folders = getFileList(baseDir);

for (i = 0; i < folders.length; i++) {

    folderName = folders[i];
    if (endsWith(folderName, "/") || endsWith(folderName, "\\"))
        folderName = substring(folderName, 0, lengthOf(folderName) - 1);

    folderPath = normPath(baseDir + folderName);

    if (!File.isDirectory(folderPath)) {
        print("Skip (not dir): " + folderPath);
        continue;
    }

    print("==== Folder: " + folderName + " ====");

    origID = openStackFromFolderOrSubfolders(folderPath, folderName);

    if (origID == -1) {
        print("Skip (no JPG/JPEG here or in subfolders): " + folderName);
        continue;
    }

    // ---- 0) 去掉最后的 _000000 ----
    p3 = lastIndexOf(folderName, "_");
    if (p3 < 0) {
        print("Skip (bad folder name): " + folderName);
        selectImage(origID); close();
        continue;
    }
    core = substring(folderName, 0, p3);

    // ---- 1) swimID：最后一个 "_" 后 ----
    p2 = lastIndexOf(core, "_");
    if (p2 < 0) {
        print("Skip (bad folder name): " + folderName);
        selectImage(origID); close();
        continue;
    }
    swimID = substring(core, p2 + 1);

    // ---- 2) 去掉最后的 "_swimID"，得到 ...wt1-MTZ-6 ----
    core2 = substring(core, 0, p2);

    // ---- 3) fishID：最后一个 "-" 后 ----
    pLastDash = lastIndexOf(core2, "-");
    if (pLastDash < 0) {
        print("Skip (bad folder name): " + folderName);
        selectImage(origID); close();
        continue;
    }
    fishID = substring(core2, pLastDash + 1);

    // ---- 4) baseCond：日期后 "_" 到最后一个 "-" 之前 ----
    pDate = indexOf(core2, "_");
    if (pDate < 0) {
        print("Skip (bad folder name): " + folderName);
        selectImage(origID); close();
        continue;
    }
    baseCond = substring(core2, pDate + 1, pLastDash);

    // ---- 5) 最终标题 ----
    newTitle = baseCond + "-" + fishID + "_" + swimID;

    // ---- 6) Scale 并 create ----
    run("Scale...", "x=- y=- z=1.0 width=1280 height=960 interpolation=Bilinear average process create title=[" + newTitle + "]");

    // 关闭原始序列窗口
    selectImage(origID);
    close();

    // ---- 7) 导出 AVI ----
    scaledTitle = getTitle();
    selectImage(scaledTitle);

    outPath = outputDir + newTitle + ".avi";
    run("AVI... ", "compression=JPEG frame=750 save=[" + outPath + "]");

    close();
}

setBatchMode(false);
print("DONE.");