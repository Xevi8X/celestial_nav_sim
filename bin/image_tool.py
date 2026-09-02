import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage,QPixmap,QPainter,QPen
from PyQt5.QtCore import Qt,QPoint,pyqtSignal


class ImageView(QWidget):
    clicked=pyqtSignal(float,float)

    def __init__(self):
        super().__init__()
        self.img=None
        self.scale=.25
        self.markers=[]
        self.links=[]
        self.mode="normal"
        self.radius=50
        self.line_width=2
        self.pending=None
        self.ox=0
        self.oy=0
        self.display_w=0
        self.display_h=0
        self.setMinimumSize(400,300)
        self.setStyleSheet("background:#111")
        self.setMouseTracking(True)

    def set_image(self,img):
        self.img=img
        self.update_geometry()
        self.update()

    def set_scale(self,s):
        self.scale=s
        self.update_geometry()
        self.update()

    def set_mode(self,mode):
        self.mode=mode
        self.pending=None
        self.update()

    def update_geometry(self):
        if self.img is None:
            return

        h,w=self.img.shape
        self.display_w=max(1,round(w*self.scale))
        self.display_h=max(1,round(h*self.scale))
        self.ox=(self.width()-self.display_w)/2
        self.oy=(self.height()-self.display_h)/2

    def resizeEvent(self,e):
        self.update_geometry()
        super().resizeEvent(e)

    def image_to_widget(self,x,y):
        return (
            self.ox+x*self.display_w/self.img.shape[1],
            self.oy+y*self.display_h/self.img.shape[0]
        )

    def widget_to_image(self,x,y):
        if self.img is None:
            return None

        if not (
            self.ox<=x<self.ox+self.display_w and
            self.oy<=y<self.oy+self.display_h
        ):
            return None

        h,w=self.img.shape

        return (
            (x-self.ox)*w/self.display_w,
            (y-self.oy)*h/self.display_h
        )

    def marker_at(self,x,y):
        r=self.radius
        for i,(mx,my) in enumerate(self.markers):
            if (x-mx)**2+(y-my)**2<=r*r:
                return i
        return None

    def annotation_color(self):
        return Qt.black if self.img.mean()>32767 else Qt.white

    def line_endpoints(self,x1,y1,x2,y2):
        dx=x2-x1
        dy=y2-y1
        d=np.hypot(dx,dy)

        if d<=2*self.radius:
            return None

        ux=dx/d
        uy=dy/d

        return (
            x1+ux*self.radius,
            y1+uy*self.radius,
            x2-ux*self.radius,
            y2-uy*self.radius
        )

    def mousePressEvent(self,e):
        if e.button()!=Qt.LeftButton or self.img is None:
            return

        p=self.widget_to_image(e.x(),e.y())

        if p is None:
            return

        x,y=p

        if self.mode=="marking":
            self.markers.append((x,y))

        elif self.mode=="connecting":
            i=self.marker_at(x,y)

            if i is None:
                return

            if self.pending is None:
                self.pending=i
            elif i!=self.pending:
                if (
                    (self.pending,i) not in self.links and
                    (i,self.pending) not in self.links
                ):
                    self.links.append((self.pending,i))
                self.pending=None

        self.update()
        self.clicked.emit(x,y)

    def paintEvent(self,e):
        if self.img is None:
            return

        p=QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        preview=(self.img>>8).astype(np.uint8)
        h,w=preview.shape

        q=QImage(
            preview.data,
            w,h,
            preview.strides[0],
            QImage.Format_Grayscale8
        ).copy()

        pix=QPixmap.fromImage(q).scaled(
            self.display_w,
            self.display_h,
            Qt.IgnoreAspectRatio,
            Qt.FastTransformation
        )

        p.drawPixmap(
            int(self.ox),
            int(self.oy),
            pix
        )

        color=self.annotation_color()
        pen=QPen(color,self.line_width)
        p.setPen(pen)

        for a,b in self.links:
            x1,y1=self.markers[a]
            x2,y2=self.markers[b]

            ep=self.line_endpoints(
                x1,y1,x2,y2
            )

            if ep is None:
                continue

            sx,sy,ex,ey=ep

            a1=self.image_to_widget(sx,sy)
            b1=self.image_to_widget(ex,ey)

            p.drawLine(
                QPoint(round(a1[0]),round(a1[1])),
                QPoint(round(b1[0]),round(b1[1]))
            )

        for i,(x,y) in enumerate(self.markers):
            q=self.image_to_widget(x,y)

            r=self.radius*self.display_w/w

            p.setPen(pen)

            p.drawEllipse(
                QPoint(round(q[0]),round(q[1])),
                round(r),
                round(r)
            )

            if i==self.pending:
                p.setPen(QPen(Qt.red,self.line_width))

                p.drawEllipse(
                    QPoint(round(q[0]),round(q[1])),
                    round(r)+3,
                    round(r)+3
                )


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Star Image Processor")
        self.resize(1200,900)

        self.img=None
        self.view=ImageView()

        self.scale=QComboBox()
        self.scale.addItems([
            "10%","25%","50%","75%","100%"
        ])
        self.scale.setCurrentText("25%")
        self.scale.currentTextChanged.connect(
            lambda x:self.view.set_scale(float(x[:-1])/100)
        )

        self.cdf=QDoubleSpinBox()
        self.cdf.setRange(0,100)
        self.cdf.setDecimals(3)
        self.cdf.setValue(99)
        self.cdf.setSuffix(" %")

        self.min_size=QSpinBox()
        self.min_size.setRange(1,1000000)
        self.min_size.setValue(18)
        self.min_size.setSuffix(" px")

        self.radius=QSpinBox()
        self.radius.setRange(1,10000)
        self.radius.setValue(50)
        self.radius.setSuffix(" px")
        self.radius.valueChanged.connect(self.set_marker_radius)

        self.line_width=QSpinBox()
        self.line_width.setRange(1,100)
        self.line_width.setValue(2)
        self.line_width.setSuffix(" px")
        self.line_width.valueChanged.connect(self.set_line_width)

        top=QHBoxLayout()

        for text,func in [
            ("Open",self.open),
            ("Save JPG",self.save),
            ("Normalize",self.normalize),
            ("Inverse",self.inverse),
            ("Threshold",self.threshold)
        ]:
            b=QPushButton(text)
            b.clicked.connect(func)
            top.addWidget(b)

        top.addSpacing(10)

        self.normal=QPushButton("Normal")
        self.mark=QPushButton("Mark")
        self.connect=QPushButton("Connect")

        for b in (
            self.normal,
            self.mark,
            self.connect
        ):
            b.setCheckable(True)

        self.normal.setChecked(True)

        group=QButtonGroup(self)
        group.setExclusive(True)

        for b in (
            self.normal,
            self.mark,
            self.connect
        ):
            group.addButton(b)
            top.addWidget(b)

        self.normal.clicked.connect(
            lambda:self.set_mode("normal")
        )

        self.mark.clicked.connect(
            lambda:self.set_mode("marking")
        )

        self.connect.clicked.connect(
            lambda:self.set_mode("connecting")
        )

        top.addSpacing(10)

        top.addWidget(QLabel("CDF"))
        top.addWidget(self.cdf)

        top.addWidget(QLabel("Min"))
        top.addWidget(self.min_size)

        top.addWidget(QLabel("Radius"))
        top.addWidget(self.radius)

        top.addWidget(QLabel("Line"))
        top.addWidget(self.line_width)

        top.addWidget(QLabel("Preview"))
        top.addWidget(self.scale)

        top.addStretch()

        self.status=QLabel()

        root=QWidget()
        layout=QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(self.view,1)
        layout.addWidget(self.status)

        self.setCentralWidget(root)
        self.update_status()

    def set_mode(self,mode):
        self.view.set_mode(mode)
        self.update_status()

    def set_marker_radius(self,v):
        self.view.radius=v
        self.update_status()
        self.view.update()

    def set_line_width(self,v):
        self.view.line_width=v
        self.update_status()
        self.view.update()

    def update_status(self):
        self.status.setText(
            f"Mode: {self.view.mode}    "
            f"Markers: {len(self.view.markers)}    "
            f"Links: {len(self.view.links)}    "
            f"Radius: {self.view.radius} px    "
            f"Line: {self.view.line_width} px"
        )

    def open(self):
        f,_=QFileDialog.getOpenFileName(
            self,
            "Open PNG",
            "",
            "PNG (*.png)"
        )

        if not f:
            return

        img=cv2.imread(
            f,
            cv2.IMREAD_UNCHANGED
        )

        if img is None:
            QMessageBox.critical(
                self,
                "Error",
                "Could not load image"
            )
            return

        if img.ndim!=2 or img.dtype!=np.uint16:
            QMessageBox.critical(
                self,
                "Error",
                f"Expected 16-bit mono PNG, got {img.dtype}"
            )
            return

        self.img=img
        self.view.markers=[]
        self.view.links=[]
        self.view.pending=None
        self.view.set_image(img)
        self.update_status()

    def save(self):
        if self.img is None:
            return

        f,_=QFileDialog.getSaveFileName(
            self,
            "Save JPG",
            "",
            "JPEG (*.jpg *.jpeg)"
        )

        if not f:
            return

        if not f.lower().endswith((".jpg",".jpeg")):
            f+=".jpg"

        out=(self.img>>8).astype(np.uint8)
        h,w=out.shape

        color=0 if self.img.mean()>32767 else 255

        for a,b in self.view.links:
            x1,y1=self.view.markers[a]
            x2,y2=self.view.markers[b]

            x1=int(round(x1))
            y1=int(round(y1))
            x2=int(round(x2))
            y2=int(round(y2))

            dx=x2-x1
            dy=y2-y1
            d=np.hypot(dx,dy)

            if d<=2*self.view.radius:
                continue

            ux=dx/d
            uy=dy/d

            sx=int(round(
                x1+ux*self.view.radius
            ))
            sy=int(round(
                y1+uy*self.view.radius
            ))
            ex=int(round(
                x2-ux*self.view.radius
            ))
            ey=int(round(
                y2-uy*self.view.radius
            ))

            cv2.line(
                out,
                (sx,sy),
                (ex,ey),
                color,
                self.view.line_width,
                cv2.LINE_AA
            )

        for x,y in self.view.markers:
            x=int(round(x))
            y=int(round(y))

            if not(0<=x<w and 0<=y<h):
                continue

            cv2.circle(
                out,
                (x,y),
                self.view.radius,
                color,
                self.view.line_width,
                cv2.LINE_AA
            )

        if not cv2.imwrite(
            f,
            out,
            [cv2.IMWRITE_JPEG_QUALITY,95]
        ):
            QMessageBox.critical(
                self,
                "Error",
                f"Could not save image:\n{f}"
            )

    def normalize(self):
        if self.img is None:
            return

        a=self.img.astype(np.float32)
        lo,hi=np.percentile(a,[1,99])

        if hi<=lo:
            return

        self.img=np.clip(
            (a-lo)*65535/(hi-lo),
            0,
            65535
        ).astype(np.uint16)

        self.view.set_image(self.img)

    def inverse(self):
        if self.img is not None:
            self.img=65535-self.img
            self.view.set_image(self.img)

    def threshold(self):
        if self.img is None:
            return

        t=np.percentile(
            self.img,
            self.cdf.value()
        )

        binary=(self.img>=t).astype(np.uint8)

        n,labels,stats,_=cv2.connectedComponentsWithStats(
            binary,8
        )

        keep=(
            stats[:,cv2.CC_STAT_AREA]>=self.min_size.value()
        ).astype(np.uint8)

        keep[0]=0

        self.img=(
            self.img*keep[labels]
        ).astype(np.uint16)

        self.view.set_image(self.img)


app=QApplication(sys.argv)
w=App()
w.show()
sys.exit(app.exec_())